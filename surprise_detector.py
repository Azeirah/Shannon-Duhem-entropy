import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, color
from scipy.ndimage import uniform_filter
from PIL import Image
import sys
import os
import random
import string

def calculate_color_entropy(image, window_size=9):
    """
    Calculate entropy considering color relationships in LAB color space.
    """
    # Convert to LAB color space
    lab_image = color.rgb2lab(image)
    
    # Calculate entropy for each channel
    entropies = []
    for channel in range(3):
        channel_data = lab_image[:, :, channel]
        # Normalize channel data to 0-1
        channel_norm = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-10)
        entropy = filters.rank.entropy(np.uint8(channel_norm * 255), np.ones((window_size, window_size)))
        entropies.append(entropy)
    
    # Weight the channels
    weighted_entropy = (0.5 * entropies[0] + 0.25 * entropies[1] + 0.25 * entropies[2])
    return weighted_entropy

def local_std(img, window_size):
    """
    Calculate local standard deviation using uniform filter.
    """
    c1 = uniform_filter(img, window_size, mode='reflect')
    c2 = uniform_filter(img*img, window_size, mode='reflect')
    return np.sqrt(abs(c2 - c1*c1))

def calculate_color_contrast(image, window_size=9):
    """
    Calculate local color contrast in LAB space using custom local std.
    """
    lab_image = color.rgb2lab(image)
    
    # Calculate local contrast for each channel
    contrasts = []
    for channel in range(3):
        channel_data = lab_image[:, :, channel]
        # Normalize channel data
        channel_norm = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-10)
        local_std_val = local_std(channel_norm, window_size)
        contrasts.append(local_std_val)
    
    # Weight the contrasts
    weighted_contrast = (0.5 * contrasts[0] + 0.25 * contrasts[1] + 0.25 * contrasts[2])
    return weighted_contrast

def calculate_local_surprise(image, window_size=9, duhem_constant=1.0):
    """
    Calculate surprise map using color entropy, contrast, and Duhem's law.
    """
    # Calculate color-aware measures
    entropy = calculate_color_entropy(image, window_size)
    contrast = calculate_color_contrast(image, window_size)
    
    # Combine entropy and contrast using Duhem's law
    surprise_map = duhem_constant * (entropy * contrast) / (1 + np.exp(-entropy))
    
    # Normalize to 0-1 range
    surprise_map = (surprise_map - surprise_map.min()) / (
        surprise_map.max() - surprise_map.min() + 1e-10
    )
    return surprise_map
    
def generate_random_hash(length=8):
    """Generate a random hash of specified length."""
    return ''.join(random.choices(string.hexdigits, k=length))

def ensure_output_directory():
    """Create outputs directory if it doesn't exist."""
    os.makedirs('outputs', exist_ok=True)

def visualize_surprise(image_path, output_path, save_individual=True):
    """Generate and save a heatmap of surprising regions with original image comparison."""
    # Load image
    image = np.array(Image.open(image_path).convert("RGB")) / 255.0
    
    # Calculate surprise map
    surprise_map = calculate_local_surprise(image)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Create separate figures for different colormaps if save_individual is True
    if save_individual:
        ensure_output_directory()
        original_name = os.path.splitext(os.path.basename(image_path))[0]
        
        colormaps = ['viridis', 'hot', 'coolwarm']
        for cmap in colormaps:
            # Create individual figure
            plt.figure(figsize=(10, 10))
            plt.imshow(surprise_map, cmap=cmap)
            plt.axis('off')
            plt.colorbar(label="Surprise Intensity")
            
            # Generate output filename
            random_hash = generate_random_hash()
            output_filename = f"{original_name}_{cmap}_{random_hash}.png"
            output_filepath = os.path.join('outputs', output_filename)
            
            # Save and close individual figure
            plt.savefig(output_filepath, bbox_inches='tight')
            plt.close()
    
    # Main visualization
    surprise_plot = ax2.imshow(surprise_map, cmap="viridis")
    ax2.set_title("Image Surprise Map")
    ax2.axis('off')
    
    # Add colorbar
    plt.colorbar(surprise_plot, ax=ax2, label="Surprise Intensity")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return surprise_map

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    output_path = img_path.rsplit('.', 1)[0] + "_surprise_map.png"
    visualize_surprise(img_path, output_path)