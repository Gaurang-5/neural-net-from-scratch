"""
Optimization algorithms.
"""

import numpy as np

class SGD:
    """
    Standard Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, learning_rate: float = 0.01):
        """
        Initializes the SGD optimizer.
        
        Parameters:
            learning_rate: The step size used for updating the weights.
        """
        self.learning_rate = learning_rate

    def update(self, layer) -> None:
        """
        Updates the parameters (weights and biases) of a given layer.
        
        Formula:
        W = W - learning_rate * dW
        b = b - learning_rate * db
        """
        layer.W = layer.W - self.learning_rate * layer.dW
        layer.b = layer.b - self.learning_rate * layer.db

class SGDMomentum:
    """
    Stochastic Gradient Descent optimizer with Momentum.
    """
    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9):
        """
        Initializes the SGDMomentum optimizer.
        
        Parameters:
            learning_rate: The step size used for updating the weights.
            beta: The momentum coefficient (usually between 0.8 and 0.99).
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity_W = {}
        self.velocity_b = {}

    def update(self, layer, layer_id: int) -> None:
        """
        Updates the parameters of a layer using momentum.
        
        Parameters:
            layer: The neural network layer to update.
            layer_id: A unique identifier for the layer to track its velocity.
        """
        # WHAT MOMENTUM DOES INTUITIVELY:
        # Think of momentum like a ball rolling down a hill. Without momentum (standard SGD),
        # the ball only looks at the current slope and can easily get stuck in small dips
        # or bounce erratically side-to-side in ravines. 
        # With momentum, the ball builds up speed in consistent directions and smooths out
        # the noisy oscillations. It effectively computes a moving average of past gradients,
        # accelerating convergence in the right direction while dampening erratic updates.
        
        if layer_id not in self.velocity_W:
            self.velocity_W[layer_id] = np.zeros_like(layer.W)
            self.velocity_b[layer_id] = np.zeros_like(layer.b)
            
        # Update velocities
        self.velocity_W[layer_id] = self.beta * self.velocity_W[layer_id] + (1 - self.beta) * layer.dW
        self.velocity_b[layer_id] = self.beta * self.velocity_b[layer_id] + (1 - self.beta) * layer.db
        
        # Update weights
        layer.W = layer.W - self.learning_rate * self.velocity_W[layer_id]
        layer.b = layer.b - self.learning_rate * self.velocity_b[layer_id]

if __name__ == "__main__":
    # Create a fake layer class to mock a DenseLayer
    class FakeLayer:
        def __init__(self):
            # Start with weights of 1.0 and biases of 0.0
            self.W = np.ones((2, 2))
            self.b = np.zeros((2, 1))
            # Gradients are pushing in a constant direction
            self.dW = np.array([[0.5, 0.5], [0.5, 0.5]])
            self.db = np.array([[0.1], [0.1]])

    print("--- Testing standard SGD ---")
    sgd = SGD(learning_rate=0.1)
    layer1 = FakeLayer()
    
    print(f"Step 0 W:\\n{layer1.W}")
    for i in range(3):
        sgd.update(layer1)
        print(f"Step {i+1} W:\\n{layer1.W}")
        
    print("\\n--- Testing SGD with Momentum ---")
    sgdm = SGDMomentum(learning_rate=0.1, beta=0.9)
    layer2 = FakeLayer()
    
    print(f"Step 0 W:\\n{layer2.W}")
    for i in range(3):
        sgdm.update(layer2, layer_id=1)
        print(f"Step {i+1} W:\\n{layer2.W}")
