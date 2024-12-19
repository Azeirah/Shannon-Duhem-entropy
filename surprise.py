from scipy.ndimage import uniform_filter
import numpy as np
from scipy.stats import entropy as calc_entropy
from sklearn.neighbors import KernelDensity

def calculate_histogram_entropy(patch, bins=256):
    """Calculate Shannon entropy using histogram method."""
    hist, _ = np.histogram(patch, bins=bins, density=True)
    # Remove zero probabilities to avoid log(0)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calculate_kde_entropy(patch, bandwidth=0.1):
    """Calculate Shannon entropy using KDE method."""
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(patch.reshape(-1, 1))
    # Sample points for probability estimation
    x = np.linspace(patch.min(), patch.max(), 256).reshape(-1, 1)
    log_prob = kde.score_samples(x)
    prob = np.exp(log_prob)
    # Normalize probabilities
    prob = prob / np.sum(prob)
    # Remove zero probabilities
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def calculate_joint_entropy(patch, window_size=2):
    """Calculate joint entropy between neighboring pixels."""
    # Create pairs of neighboring pixels
    pairs = []
    for i in range(patch.shape[0] - 1):
        for j in range(patch.shape[1] - 1):
            pairs.append([patch[i,j], patch[i+1,j]])
            pairs.append([patch[i,j], patch[i,j+1]])
    pairs = np.array(pairs)
    
    # Calculate 2D histogram
    hist, _, _ = np.histogram2d(pairs[:,0], pairs[:,1], bins=16, density=True)
    # Remove zero probabilities
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def extract_patches(image, window_size):
    """Extract all overlapping patches from image."""
    patches = np.lib.stride_tricks.sliding_window_view(
        image, (window_size, window_size)
    )
    return patches

def calculate_local_entropy(image, window_size, method='histogram'):
    """Calculate local entropy for each pixel using specified method."""
    # Pad image to handle edges
    pad = window_size // 2
    padded = np.pad(image, pad, mode='reflect')
    
    # Initialize output
    entropy_map = np.zeros_like(image)
    
    # Extract all patches
    patches = extract_patches(padded, window_size)
    
    # Calculate entropy for each patch
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = patches[i:i+window_size, j:j+window_size]
            if method == 'histogram':
                entropy_map[i,j] = calculate_histogram_entropy(patch)
            elif method == 'kde':
                entropy_map[i,j] = calculate_kde_entropy(patch)
            elif method == 'joint':
                entropy_map[i,j] = calculate_joint_entropy(patch)
            else:
                raise ValueError(f"Unknown entropy method: {method}")
    
    return entropy_map

def local_std(img, window_size):
    """Calculate local standard deviation using uniform filter."""
    c1 = uniform_filter(img, window_size, mode='reflect')
    c2 = uniform_filter(img*img, window_size, mode='reflect')
    return np.sqrt(abs(c2 - c1*c1))

def calculate_surprise(image, window_size=5, duhem_constant=1.0, entropy_method='histogram'):
    """
    Calculate surprise map using specified entropy method.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    window_size : int
        Size of the local window for calculations
    duhem_constant : float
        Constant for Duhem's law combination
    entropy_method : str
        Method for entropy calculation: 'histogram', 'kde', or 'joint'
    
    Returns:
    --------
    ndarray
        Normalized surprise map
    """
    # Ensure image is normalized to 0-1
    image_norm = image / 255.0 if image.max() > 1 else image
    
    # Calculate local entropy using specified method
    entropy = calculate_local_entropy(image_norm, window_size, entropy_method)
    entropy_clipped = np.clip(entropy, -100, 100)  # Choose reasonable bounds
    
    # Or use a more numerically stable sigmoid implementation
    def stable_sigmoid(x):
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))
    
    # Calculate contrast
    contrast = local_std(image_norm, window_size)

    # Calculate surprise using stable method
    surprise = duhem_constant * (entropy_clipped * contrast) * stable_sigmoid(entropy_clipped)
    
    # Normalize
    surprise = (surprise - surprise.min()) / (surprise.max() - surprise.min() + 1e-10)
    return surprise
