"""
Helper functions for data processing and evaluation.
"""

import numpy as np

def load_mnist():
    """
    Loads the MNIST dataset using sklearn (bypassing the macOS TensorFlow C++ abort).
    Flattens images, normalizes pixel values, and transposes to (784, n_samples).
    """
    from sklearn.datasets import fetch_openml
    
    print("Downloading MNIST from OpenML (this may take a few seconds)...")
    # as_frame=False ensures we get numpy arrays instead of pandas DataFrames
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    
    X = mnist.data
    Y = mnist.target.astype(int)
    
    # MNIST standard split is 60,000 training and 10,000 testing
    X_train, X_test = X[:60000], X[60000:]
    Y_train, Y_test = Y[:60000], Y[60000:]
    
    # Normalize to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Transpose to (784, n_samples)
    X_train = X_train.T
    X_test = X_test.T
    
    return X_train, Y_train, X_test, Y_test

def one_hot_encode(Y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """
    Converts a 1D array of integer labels to a one-hot encoded matrix.
    Input shape: (n_samples,)
    Output shape: (n_classes, n_samples)
    """
    # Create an identity matrix of size (n_classes, n_classes).
    # Indexing it with integer array Y yields an array of shape (n_samples, n_classes).
    # Transposing it gives the desired shape (n_classes, n_samples).
    return np.eye(n_classes)[Y].T

def normalize(X: np.ndarray, mean=None, std=None):
    """
    Normalizes the input data to have zero mean and unit variance.
    X shape is assumed to be (n_features, n_samples).
    If mean and std are not provided, they are computed from X.
    """
    epsilon = 1e-8
    
    if mean is None:
        mean = np.mean(X, axis=1, keepdims=True)
    if std is None:
        std = np.std(X, axis=1, keepdims=True)
        
    X_norm = (X - mean) / (std + epsilon)
    
    return X_norm, mean, std

def train_val_split(X: np.ndarray, Y: np.ndarray, val_size: float = 0.1, seed: int = 42):
    """
    Splits the data into training and validation sets after shuffling.
    X shape: (n_features, n_samples)
    Y shape: can be (n_samples,) or (n_classes, n_samples)
    """
    np.random.seed(seed)
    n_samples = X.shape[1]
    
    permutation = np.random.permutation(n_samples)
    
    # Shuffle X
    X_shuffled = X[:, permutation]
    
    # Shuffle Y correctly based on whether it is 1D or 2D
    if len(Y.shape) == 1:
        Y_shuffled = Y[permutation]
    else:
        Y_shuffled = Y[:, permutation]
        
    val_count = int(n_samples * val_size)
    
    # Split X
    X_val = X_shuffled[:, :val_count]
    X_train = X_shuffled[:, val_count:]
    
    # Split Y
    if len(Y.shape) == 1:
        Y_val = Y_shuffled[:val_count]
        Y_train = Y_shuffled[val_count:]
    else:
        Y_val = Y_shuffled[:, :val_count]
        Y_train = Y_shuffled[:, val_count:]
        
    return X_train, X_val, Y_train, Y_val

if __name__ == "__main__":
    import os
    # Disable TF logs to keep output clean
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    print("Loading MNIST dataset...")
    try:
        X_train, Y_train, X_test, Y_test = load_mnist()
    except ImportError:
        print("tensorflow is not installed. Please install it to test load_mnist.")
        exit(1)
        
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    
    min_val = X_train.min()
    max_val = X_train.max()
    print(f"X_train min value: {min_val}")
    print(f"X_train max value: {max_val}")
    
    print("\\nFirst 5 labels:")
    print(Y_train[:5])
    
    print("\\nOne-hot encoded first 5 labels:")
    one_hot_subset = one_hot_encode(Y_train[:5], n_classes=10)
    print(one_hot_subset)
    
    # Validate correctness
    passed = True
    if X_train.shape != (784, 60000): passed = False
    if Y_train.shape != (60000,): passed = False
    if X_test.shape != (784, 10000): passed = False
    if Y_test.shape != (10000,): passed = False
    if min_val < 0.0 or max_val > 1.0: passed = False
    if one_hot_subset.shape != (10, 5): passed = False
    
    if passed:
        print("\\nAll checks PASSED")
    else:
        print("\\nSome checks FAILED")
