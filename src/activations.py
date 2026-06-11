"""
Activation functions and their derivatives.
"""

import numpy as np

def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid activation function.
    Formula: 1 / (1 + exp(-Z))
    
    In plain English: This function takes the input Z, multiplies it by negative one, 
    exponentiates it, adds one, and then takes the inverse. This 'squashes' the input 
    values into a range between 0 and 1, making it useful for probability estimation.
    """
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_derivative(Z: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the sigmoid activation function.
    Formula: sigmoid(Z) * (1 - sigmoid(Z))
    
    In plain English: This function computes the sigmoid of the input and then 
    multiplies it by one minus the sigmoid of the input. It represents the gradient 
    of the sigmoid function at a given point Z.
    """
    s = sigmoid(Z)
    return s * (1.0 - s)

def relu(Z: np.ndarray) -> np.ndarray:
    """
    Computes the Rectified Linear Unit (ReLU) activation function.
    Formula: max(0, Z) elementwise
    
    In plain English: This function returns the input value itself if the input 
    is greater than zero, and returns zero if the input is less than or equal to zero. 
    It effectively thresholds negative values to zero.
    """
    return np.maximum(0.0, Z)

def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the ReLU activation function.
    Formula: 1 where Z > 0, else 0
    
    In plain English: This function returns a value of 1 for any input that is 
    strictly greater than zero, and 0 for any input that is less than or equal to zero.
    """
    return (Z > 0).astype(float)

def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Computes the softmax activation function.
    Formula: exp(Z - max(Z)) / sum(exp(Z - max(Z)))
    
    In plain English: It first subtracts the maximum value in the array from all 
    elements (along the class dimension) to prevent numerical overflow. Then, it 
    exponentiates each resulting value and divides by the sum of these exponentials, 
    creating a probability distribution that sums to 1.
    """
    # Z shape is expected to be (n_classes, batch_size)
    # Subtract max for numerical stability along the class axis (axis 0)
    shift_Z = Z - np.max(Z, axis=0, keepdims=True)
    exps = np.exp(shift_Z)
    return exps / np.sum(exps, axis=0, keepdims=True)

if __name__ == "__main__":
    # Test sigmoid
    Z_test = np.array([0.0])
    if np.allclose(sigmoid(Z_test), np.array([0.5])):
        print("sigmoid: PASS")
    else:
        print("sigmoid: FAIL")

    # Test sigmoid_derivative
    if np.allclose(sigmoid_derivative(Z_test), np.array([0.25])):
        print("sigmoid_derivative: PASS")
    else:
        print("sigmoid_derivative: FAIL")

    # Test relu
    Z_test_relu = np.array([-1.0, 0.0, 2.0])
    if np.allclose(relu(Z_test_relu), np.array([0.0, 0.0, 2.0])):
        print("relu: PASS")
    else:
        print("relu: FAIL")

    # Test relu_derivative
    if np.allclose(relu_derivative(Z_test_relu), np.array([0.0, 0.0, 1.0])):
        print("relu_derivative: PASS")
    else:
        print("relu_derivative: FAIL")

    # Test softmax
    # Shape: (n_classes=2, batch_size=2)
    Z_test_softmax = np.array([[1.0, 2.0], 
                               [1.0, 2.0]]) 
    expected_softmax = np.array([[0.5, 0.5], 
                                 [0.5, 0.5]])
    if np.allclose(softmax(Z_test_softmax), expected_softmax):
        print("softmax: PASS")
    else:
        print("softmax: FAIL")
