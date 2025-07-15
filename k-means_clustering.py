import torch
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from torch import nn
from trainVGG import OptimizedVGG16  # 导入 OptimizedVGG16 类
import time

class VGG16Compressor:
    def __init__(self, num_classes=10):
        # Build the structure of the OptimizedVGG16 model
        self.original_model = OptimizedVGG16()
        # Load the saved model parameters
        try:
            self.original_model.load_state_dict(torch.load('./final_model.pth'))
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model file not found. Please check the path.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

        self.model = self.original_model
        self.model.eval()
        self.device = torch.device("cpu")
        self.num_classes = num_classes

    def extract_3x3_kernels(self):
        start_time = time.time()
        single_kernels = []
        layer_info = []
        conv_layer_indices = []
        original_shapes = []
        # Extract convolutional kernels from the original model
        for m, layer in enumerate(self.original_model.features):
            if isinstance(layer, torch.nn.Conv2d) and layer.kernel_size == (3, 3):
                weights = layer.weight.data.cpu().numpy()
                out_ch, in_ch, h, w = weights.shape
                layer_info.append({
                    "out_ch": out_ch,
                    "in_ch": in_ch,
                    "stride": layer.stride,
                    "padding": layer.padding,
                    "bias": layer.bias is not None,
                    "dilation": layer.dilation,
                    "groups": layer.groups,
                    "padding_mode": layer.padding_mode
                })

                for j in range(out_ch):
                    for i in range(in_ch):
                        single_kernel = weights[j, i].reshape(1, -1)
                        single_kernels.append(single_kernel)
                        original_shapes.append((h, w))

                conv_layer_indices.append(m)

        single_kernels = np.vstack(single_kernels)

        print("\nThe first three single kernels of the first layer:")
        for i in range(min(3, len(single_kernels))):
            print(f"  Single kernel {i+1}:")
            print(single_kernels[i].reshape(3, 3))
            print()
        print(f"Shape of single kernels: {single_kernels.shape}")
        end_time = time.time()
        print(f"extract_3x3_kernels took {end_time - start_time:.4f} seconds")
        return single_kernels, layer_info, conv_layer_indices, original_shapes

    def calculate_scales(self, single_kernels):
        start_time = time.time()
        scales = []
        center_idx = (3 * 3) // 2

        for kernel in single_kernels:
            w_center = kernel[center_idx]
            norm = np.linalg.norm(kernel)
            scale = np.sign(w_center) * norm 
            scales.append(scale)

        print("\nScales of the first three kernels of the first layer:")
        for i in range(min(3, len(scales))):
            print(f"Kernel {i+1}: scale = {scales[i]:.4f}")

        scales_array = np.array(scales, dtype=np.float16)
        print(f"Shape of scales: {scales_array.shape}")
        end_time = time.time()
        print(f"calculate_scales took {end_time - start_time:.4f} seconds")
        return np.array(scales)

    def cluster_kernels(self, single_kernels, scales, k=128, normalize=True, n_init=50):
        start_time = time.time()
        # Calculate the scaling factor for each convolutional kernel
        if normalize:
            # Ensure the shape of the scaling factors is correct (n,)
            scales = np.array(scales).flatten()

            # Avoid division by zero or very small values
            min_scale = 1e-8
            safe_scales = np.maximum(np.abs(scales), min_scale)

            # Get the signs of the scales
            signs = np.sign(scales)
            # Normalize the kernels
            normalized_kernels = single_kernels / (signs[:, np.newaxis] * safe_scales[:, np.newaxis])
        else:
            normalized_kernels = single_kernels

        print(f"\nShape of normalized_kernels: {normalized_kernels.shape}")
        print("\nThe first three normalized kernels of the first layer:")
        for i in range(min(3, len(normalized_kernels))):
            # Reshape the 1D array to a 3x3 matrix and print
            print(f"Normalized kernel {i+1} =")
            print(normalized_kernels[i].reshape(3, 3))
            print()

        # Perform K-means clustering
        print(f"Performing K-means clustering (k={k}, n_init={n_init})...")
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        # Fit the model and predict the cluster labels
        cluster_labels = kmeans.fit_predict(normalized_kernels)
        # Get the cluster centers
        centroids = kmeans.cluster_centers_
        print(f"\nShape of centroids: {centroids.shape}")
        # Count the size of each cluster
        cluster_sizes = np.bincount(cluster_labels, minlength=k)

        # Build cluster statistics
        cluster_stats = {
            "sizes": cluster_sizes,
            "min_size": np.min(cluster_sizes),
            "max_size": np.max(cluster_sizes),
            "empty_clusters": np.sum(cluster_sizes == 0)
        }

        print(f"Clustering completed! Minimum/Maximum cluster size: {cluster_stats['min_size']}/{cluster_stats['max_size']}")
        print(f"Number of empty clusters: {cluster_stats['empty_clusters']}")
        end_time = time.time()
        print(f"cluster_kernels took {end_time - start_time:.4f} seconds")
        return cluster_labels, centroids, cluster_stats

    def build_compressed_model(self, cluster_labels, centroids, scales, layer_info, conv_layer_indices, original_shapes):
        start_time = time.time()
        """Build a compressed model by reconstructing convolutional kernels through traversal"""
        compressed_model = OptimizedVGG16()
        compressed_features = list(compressed_model.features.children())
        idx_ptr = 0
        m = 0

        for m, layer in enumerate(self.original_model.features):
            if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
                in_ch = layer.in_channels
                out_ch = layer.out_channels
                layer_idx = conv_layer_indices.index(m)
                group_info = layer_info[layer_idx]

                # Create a new convolutional layer
                new_conv = nn.Conv2d(
                    in_channels=group_info["in_ch"],
                    out_channels=group_info["out_ch"],
                    kernel_size=3,
                    stride=group_info["stride"],
                    padding=group_info["padding"],
                    bias=group_info["bias"],
                    dilation=group_info["dilation"],
                    groups=group_info["groups"],
                    padding_mode=group_info["padding_mode"]
                )

                if group_info["bias"]:
                   new_conv.bias.data = layer.bias.data.clone()

                new_weights = []
                if m == 0:
                    print("\nFirst layer: Centroids and scales of the first three compressed kernels:")
                    for i in range(min(3, in_ch)):
                        centroid = centroids[cluster_labels[idx_ptr]]
                        scale = scales[idx_ptr]
                        print(f"  Kernel {i+1}:")
                        print(f"  Centroid:\n{centroid.reshape(3, 3)}")
                        print(f"  Scale: {scale:.4f}")
                        idx_ptr += 1
                    idx_ptr = 0  # Reset idx_ptr for the normal kernel reconstruction

                for j in range(out_ch):
                    channel_weights = []
                    for i in range(in_ch):
                        centroid = centroids[cluster_labels[idx_ptr]]
                        scale = scales[idx_ptr]
                        kernel = centroid.reshape(*original_shapes[idx_ptr]) * scale
                        channel_weights.append(kernel)
                        idx_ptr += 1
                    channel_weights = np.stack(channel_weights, axis=0)
                    new_weights.append(channel_weights)

                new_weights = np.stack(new_weights, axis=0)
                #print(f"\nShape of new_weights: {new_weights.shape}")
                new_conv.weight.data = torch.tensor(new_weights, dtype=torch.float32)

                if m == 0:
                    print(f"\nLayer {m}: The first three reconstructed kernels after compression")
                    for j in range(min(1, out_ch)):
                        for i in range(min(3, in_ch)):
                            print(f"  Kernel [{j}, {i}]:")
                            print(new_weights[j, i])

                compressed_features[m] = new_conv
                m += 1
            else:
                compressed_features[m] = layer
                m += 1

        compressed_model.features = nn.Sequential(*compressed_features)
        end_time = time.time()
        print(f"build_compressed_model took {end_time - start_time:.4f} seconds")
        return compressed_model

    def save_compressed_model(self, compressed_model, save_path):
        start_time = time.time()
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(compressed_model.state_dict(), save_path)
        print(f"Compressed model saved to: {save_path}")
        end_time = time.time()
        print(f"save_compressed_model took {end_time - start_time:.4f} seconds")

    def calculate_compression_ratio(self, cluster_labels, centroids):
        """Calculate the compression ratio"""
        N = len(cluster_labels)
        k = len(centroids)
        bs = 16
        bw = 32
        h, w = 3, 3

        # Calculate the number of bits for the original storage
        original_bits = N * bw * h * w

        # Calculate the number of bits for the codebook
        codebook_bits = k * bw * h * w
        # Calculate the number of bits for the indices
        indices_bits = N * np.ceil(np.log2(k))
        # Calculate the number of bits for the scales
        scales_bits = N * bs

        # Total number of bits after compression
        compressed_bits = indices_bits + scales_bits + codebook_bits

        # Calculate the compression ratio
        compression_ratio = original_bits / compressed_bits

        return compression_ratio

    def save_compressed_model_dynamic(self, centroids, cluster_labels, scales, layer_info, conv_layer_indices, save_path):
        start_time = time.time()
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save({
            "centroids": centroids,
            "labels": cluster_labels,
            "scales": scales,
            "layer_info": layer_info,
            "conv_layer_indices": conv_layer_indices,
            "classifier": self.model.classifier.state_dict(),
            "num_classes": self.num_classes
        }, save_path)

        compression_ratio = self.calculate_compression_ratio(cluster_labels, centroids)

        print(f"Dynamic model saved to: {save_path}")
        print(f"Number of kernels: {len(cluster_labels)}, Number of centroids: {len(centroids)}")
        print(f"Compression rate: {compression_ratio:.2f}x")
        end_time = time.time()
        print(f"save_compressed_model_dynamic took {end_time - start_time:.4f} seconds")

def format_time(self, seconds):
        """将秒转换为时分秒格式"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

# Run the code
if __name__ == "__main__":
    start_time = time.time()
    # Initialize an instance of the VGG16Compressor class
    compressor = VGG16Compressor(num_classes=10)

    # Extract 3x3 convolutional kernels
    single_kernels, layer_info, conv_layer_indices, original_shapes = compressor.extract_3x3_kernels()

    # Calculate the scaling factor for each convolutional kernel
    scales = compressor.calculate_scales(single_kernels)

    # Change the value of n_init here
    n_init_value = 100
    # Perform K-means clustering
    cluster_labels, centroids, cluster_stats = compressor.cluster_kernels(single_kernels, scales, k=8192, n_init=10)

    # Build the compressed model
    compressed_model = compressor.build_compressed_model(cluster_labels, centroids, scales, layer_info, conv_layer_indices, original_shapes)

    # Save the compressed model
    compressor.save_compressed_model(compressed_model, save_path="./compressed_models/vgg16_8192_full.pth")
    # Save the dynamic compressed model
    compressor.save_compressed_model_dynamic(centroids, cluster_labels, scales, layer_info, conv_layer_indices, save_path="./compressed_models/vgg16_8192.pth")

    end_time = time.time()
    total_elapsed = end_time - start_time
    print(f"Total running time: {compressor.format_time(total_elapsed)}")
