"""
Neural network layers.
"""

import numpy as np
from src.activations import sigmoid, sigmoid_derivative, relu, relu_derivative, softmax

class DenseLayer:
    """
    A fully connected (dense) neural network layer.
    """
    
    def __init__(self, n_inputs: int, n_neurons: int, activation: str = 'relu'):
        """
        Initializes the DenseLayer.
        
        Parameters:
            n_inputs: number of input features
            n_neurons: number of neurons in this layer
            activation: string, either 'relu', 'sigmoid', or 'softmax'
            
        In plain English: This method sets up the initial weights and biases for the layer.
        The weights are randomized, and the biases start at zero. It also prepares
        caches to store inputs and pre-activation values during the forward pass,
        which are necessary for computing gradients in the backward pass.
        """
        # WHY Xavier initialization helps:
        # Initializing weights with a standard normal distribution can lead to outputs
        # that grow too large or shrink too small exponentially across layers. By scaling
        # the random weights by sqrt(2.0 / n_inputs), we keep the variance of the
        # activations relatively constant across layers. This prevents the gradients
        # from vanishing (becoming too small to learn) or exploding (becoming too large
        # and unstable) during backpropagation.
        self.W = np.random.randn(n_neurons, n_inputs) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros((n_neurons, 1))
        self.activation = activation
        
        self.input_cache = None
        self.Z_cache = None
        
        self.dW = None
        self.db = None

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass for this layer.
        Formula: Z = W · A_prev + b, followed by A = activation(Z)
        
        In plain English: This method multiplies the input data by the layer's weights
        and adds the bias. This creates a linear transformation (Z). Then, it passes
        Z through a non-linear activation function to produce the layer's final output (A).
        """
        if A_prev.shape[0] != self.W.shape[1]:
            raise ValueError(
                f"Shape mismatch: Input A_prev has {A_prev.shape[0]} features, "
                f"but this layer expects {self.W.shape[1]} features."
            )
            
        Z = np.dot(self.W, A_prev) + self.b
        
        self.input_cache = A_prev
        self.Z_cache = Z
        
        if self.activation == 'relu':
            A = relu(Z)
        elif self.activation == 'sigmoid':
            A = sigmoid(Z)
        elif self.activation == 'softmax':
            A = softmax(Z)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
            
        return A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass for this layer to compute gradients.
        
        Formulas:
        dZ = dA * activation_derivative(Z)  (element-wise)
        dW = (1/batch_size) * dZ · A_prev.T
        db = (1/batch_size) * sum(dZ, axis=1)
        dA_prev = W.T · dZ
        
        In plain English: This method calculates how much the layer's weights, biases,
        and inputs contributed to the overall error. It takes the gradient of the error
        with respect to its output (dA), backpropagates it through the activation function
        to get dZ, and then uses dZ to find the gradients for the weights (dW) and biases (db).
        Finally, it passes the error backward to the previous layer by calculating dA_prev.
        """
        batch_size = self.input_cache.shape[1]
        
        if self.activation == 'relu':
            dZ = dA * relu_derivative(self.Z_cache)
        elif self.activation == 'sigmoid':
            dZ = dA * sigmoid_derivative(self.Z_cache)
        elif self.activation == 'softmax':
            # For softmax combined with cross entropy, the network passes dZ directly 
            # as the dA argument (since dZ = Y_pred - Y_true).
            dZ = dA
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
            
        self.dW = (1.0 / batch_size) * np.dot(dZ, self.input_cache.T)
        self.db = (1.0 / batch_size) * np.sum(dZ, axis=1, keepdims=True)
        
        dA_prev = np.dot(self.W.T, dZ)
        
        return dA_prev

if __name__ == "__main__":
    # Test block
    np.random.seed(42)
    
    # Create a layer with 4 inputs and 3 neurons
    layer = DenseLayer(n_inputs=4, n_neurons=3, activation='relu')
    
    # Create random (4,5) input batch
    A_prev_test = np.random.randn(4, 5)
    
    # Forward pass
    print("Running forward pass...")
    A_test = layer.forward(A_prev_test)
    
    # Backward pass
    print("Running backward pass...")
    dA_test = np.random.randn(*A_test.shape)
    dA_prev_test = layer.backward(dA_test)
    
    # Print shapes
    print(f"Input shape (A_prev): {A_prev_test.shape}")
    print(f"Weights shape (W): {layer.W.shape}")
    print(f"Biases shape (b): {layer.b.shape}")
    print(f"Output shape (A): {A_test.shape}")
    print(f"Gradient dA shape: {dA_test.shape}")
    print(f"Gradient dW shape: {layer.dW.shape}")
    print(f"Gradient db shape: {layer.db.shape}")
    print(f"Gradient dA_prev shape: {dA_prev_test.shape}")
    print("PASS")
