"""
Main NeuralNetwork class to connect layers.
"""

import numpy as np
from src.layers import DenseLayer
from src.losses import cross_entropy_loss, cross_entropy_derivative

class NeuralNetwork:
    """
    A class representing a fully connected neural network.
    """
    def __init__(self, optimizer):
        """
        Initializes the NeuralNetwork.
        
        Parameters:
            optimizer: An instance of an optimizer (like SGD or SGDMomentum).
        """
        self.optimizer = optimizer
        self.layers = []
        self.loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        self.val_accuracy_history = []

    def add_layer(self, layer: DenseLayer) -> None:
        """
        Appends a DenseLayer to the network.
        """
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the entire network.
        
        Parameters:
            X: Input data of shape (n_features, batch_size)
        Returns:
            A: The output of the final layer.
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def compute_loss(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """
        Computes the cross-entropy loss given predictions and true one-hot labels.
        """
        return cross_entropy_loss(Y_pred, Y_true)

    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> None:
        """
        Performs the backward pass through the entire network.
        Computes the initial gradient and propagates it backward through all layers.
        """
        # Compute the gradient of the loss with respect to the output
        dA = cross_entropy_derivative(Y_pred, Y_true)
        
        # Propagate backward through layers in reverse order
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update_weights(self) -> None:
        """
        Updates the weights and biases of all layers using the initialized optimizer.
        """
        for layer_id, layer in enumerate(self.layers):
            try:
                # Try passing layer_id for optimizers that track velocity per layer
                self.optimizer.update(layer, layer_id=layer_id)
            except TypeError:
                # Fallback for optimizers that only take the layer (like standard SGD)
                self.optimizer.update(layer)

    def save_weights(self, filepath: str) -> None:
        """Saves the weights and biases of all layers to a compressed .npz file."""
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f'W_{i}'] = layer.W
            weights[f'b_{i}'] = layer.b
        np.savez(filepath, **weights)

    def load_weights(self, filepath: str) -> None:
        """Loads weights and biases from a .npz file into the network layers."""
        data = np.load(filepath)
        for i, layer in enumerate(self.layers):
            layer.W = data[f'W_{i}']
            layer.b = data[f'b_{i}']

    def train_one_epoch(self, X: np.ndarray, Y: np.ndarray, batch_size: int = 32) -> float:
        """
        Trains the network for one epoch using mini-batch gradient descent.
        """
        m = X.shape[1]
        
        # Shuffle X and Y together
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]
        
        num_batches = m // batch_size
        if m % batch_size != 0:
            num_batches += 1
            
        epoch_loss = 0.0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, m)
            
            X_batch = X_shuffled[:, start_idx:end_idx]
            Y_batch = Y_shuffled[:, start_idx:end_idx]
            
            # Forward pass
            Y_pred = self.forward(X_batch)
            
            # Compute loss and weight by batch size for an accurate epoch mean
            batch_loss = self.compute_loss(Y_pred, Y_batch)
            epoch_loss += batch_loss * (end_idx - start_idx)
            
            # Backward pass
            self.backward(Y_pred, Y_batch)
            
            # Update weights
            self.update_weights()
            
        return epoch_loss / m

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the given input data.
        Returns the class index with the highest probability.
        """
        Y_pred = self.forward(X)
        return np.argmax(Y_pred, axis=0)

    def evaluate(self, X: np.ndarray, Y_true_labels: np.ndarray) -> float:
        """
        Evaluates the network's accuracy on the given data.
        Y_true_labels should be 1D integer labels (not one-hot).
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y_true_labels)
        return float(accuracy)

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, 
              X_val: np.ndarray, Y_val: np.ndarray, 
              epochs: int = 20, batch_size: int = 32, verbose: bool = True):
        """
        Trains the neural network over multiple epochs.
        Y_train and Y_val must be one-hot encoded.
        """
        # Convert one-hot Y_train and Y_val to integer labels for evaluation
        Y_train_labels = np.argmax(Y_train, axis=0)
        Y_val_labels = np.argmax(Y_val, axis=0)
        
        for epoch in range(epochs):
            # Train one epoch
            train_loss = self.train_one_epoch(X_train, Y_train, batch_size)
            
            # Compute train accuracy
            train_acc = self.evaluate(X_train, Y_train_labels)
            
            # Compute val loss and accuracy
            Y_val_pred = self.forward(X_val)
            val_loss = self.compute_loss(Y_val_pred, Y_val)
            val_acc = self.evaluate(X_val, Y_val_labels)
            
            # Append metrics to history
            self.loss_history.append(train_loss)
            self.accuracy_history.append(train_acc)
            self.val_loss_history.append(val_loss)
            self.val_accuracy_history.append(val_acc)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                
        return self

if __name__ == "__main__":
    from src.optimizers import SGDMomentum
    
    # Smoke test setup
    np.random.seed(42)
    
    # Create random data: 4 features, 20 samples
    X_test = np.random.randn(4, 20)
    
    # Create fake one-hot labels for 2 classes
    labels = np.random.randint(0, 2, 20)
    Y_test = np.zeros((2, 20))
    Y_test[labels, np.arange(20)] = 1
    
    # Initialize network with SGDMomentum
    optimizer = SGDMomentum(learning_rate=0.1, beta=0.9)
    nn = NeuralNetwork(optimizer)
    
    # Add layers (4 -> 3 -> 2)
    nn.add_layer(DenseLayer(n_inputs=4, n_neurons=3, activation='relu'))
    nn.add_layer(DenseLayer(n_inputs=3, n_neurons=2, activation='softmax'))
    
    print("Running 2 epochs smoke test...")
    nn.train(X_test, Y_test, X_test, Y_test, epochs=2, batch_size=8, verbose=True)
    print("Smoke test completed without errors!")
