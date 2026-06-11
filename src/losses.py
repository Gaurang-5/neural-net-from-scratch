"""
Loss functions and their derivatives.
"""

import numpy as np

def cross_entropy_loss(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """
    Computes the cross-entropy loss.
    Formula: -mean(sum(Y_true * log(Y_pred)))
    
    In plain English: This function takes the true one-hot encoded labels and the 
    predicted probabilities, multiplies the true labels by the natural logarithm of 
    the predicted probabilities, sums these values across all classes for each sample, 
    takes the negative of that sum, and finally computes the average loss across all 
    samples in the batch.
    """
    epsilon = 1e-8
    # Add epsilon inside the log to prevent log(0)
    loss_per_sample = -np.sum(Y_true * np.log(Y_pred + epsilon), axis=0)
    loss = np.mean(loss_per_sample)
    
    return float(loss)

def cross_entropy_derivative(Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    """
    Computes the combined derivative of the softmax activation and cross-entropy loss.
    Formula: Y_pred - Y_true
    
    In plain English: This function subtracts the true one-hot encoded labels from 
    the predicted probabilities. This yields the exact error gradient with respect to 
    the pre-activation inputs (Z) of the output layer.
    """
    # WHY THIS SIMPLIFICATION WORKS MATHEMATICALLY:
    # By the chain rule, the gradient of the loss with respect to the pre-activation Z
    # is (dL / dZ) = (dL / dA) * (dA / dZ), where A is the softmax output (Y_pred).
    # 
    # 1. The derivative of cross-entropy loss (L) with respect to softmax output (A_i) is:
    #    dL / dA_i = - Y_true_i / A_i
    #
    # 2. The derivative of softmax output (A_i) with respect to pre-activation (Z_j) is:
    #    dA_i / dZ_j = A_i * (1 - A_i)  if i == j
    #    dA_i / dZ_j = - A_i * A_j      if i != j
    #
    # When you multiply these derivatives and sum over all classes, the terms gracefully
    # cancel out. The complex Jacobian matrix multiplication collapses perfectly into
    # a simple subtraction: A - Y_true, or Y_pred - Y_true.
    
    return Y_pred - Y_true

if __name__ == "__main__":
    # Create a fake Y_pred
    # Shape: (n_classes=3, batch_size=2)
    # Probabilities sum to 1 along columns (axis=0) for this shape
    Y_pred_fake = np.array([
        [0.7, 0.2],
        [0.2, 0.7],
        [0.1, 0.1]
    ])
    
    # Create a fake Y_true (one-hot)
    # Shape: (n_classes=3, batch_size=2)
    Y_true_fake = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    
    loss_val = cross_entropy_loss(Y_pred_fake, Y_true_fake)
    print(f"Loss value: {loss_val}")
    
    if isinstance(loss_val, float) and loss_val > 0:
        print("PASS")
    else:
        print("FAIL")
