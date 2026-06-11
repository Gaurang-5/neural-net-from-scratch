"""
Training script to initialize and train the network.
"""

import os
import time
import json
import numpy as np
from sklearn.metrics import classification_report

from src.utils import load_mnist, train_val_split, one_hot_encode
from src.optimizers import SGDMomentum
from src.network import NeuralNetwork
from src.layers import DenseLayer

# Disable TF logs to keep output clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    # Seed for reproducibility
    np.random.seed(42)

    # === DATA ===
    print("Loading and preparing data...")
    X_train_full, Y_train_full, X_test, Y_test = load_mnist()
    
    # 10% validation set split
    X_train, X_val, Y_train, Y_val = train_val_split(X_train_full, Y_train_full, val_size=0.1, seed=42)
    
    # One-hot encode training and validation labels
    Y_train_oh = one_hot_encode(Y_train, n_classes=10)
    Y_val_oh = one_hot_encode(Y_val, n_classes=10)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train_oh shape: {Y_train_oh.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"Y_val_oh shape: {Y_val_oh.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    
    # === MODEL ===
    print("\\nBuilding network...")
    optimizer = SGDMomentum(learning_rate=0.01, beta=0.9)
    nn = NeuralNetwork(optimizer)
    
    # Add layers
    nn.add_layer(DenseLayer(n_inputs=784, n_neurons=128, activation='relu'))
    nn.add_layer(DenseLayer(n_inputs=128, n_neurons=64, activation='relu'))
    nn.add_layer(DenseLayer(n_inputs=64, n_neurons=10, activation='softmax'))
    
    # Print summary
    print("\\nModel Summary:")
    print("-" * 55)
    print(f"{'Layer':<15} | {'Input Shape':<12} | {'Output Shape':<12} | {'Parameters'}")
    print("-" * 55)
    
    total_params = 0
    in_features = 784
    for i, layer in enumerate(nn.layers):
        out_features = layer.W.shape[0]
        w_params = layer.W.size
        b_params = layer.b.size
        params = w_params + b_params
        total_params += params
        
        layer_name = f"DenseLayer_{i+1}"
        print(f"{layer_name:<15} | ({in_features:<10}) | ({out_features:<10}) | {params:,}")
        in_features = out_features
        
    print("-" * 55)
    print(f"Total Parameters: {total_params:,}\\n")
    
    # === TRAIN ===
    print("Starting training...")
    start_time = time.time()
    
    nn.train(X_train, Y_train_oh, X_val, Y_val_oh, epochs=20, batch_size=32, verbose=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\\nTotal training time: {total_time:.2f} seconds")
    
    # === EVALUATE ===
    print("\\nEvaluating on test set...")
    test_acc = nn.evaluate(X_test, Y_test)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    if test_acc < 0.95:
        print("WARNING: accuracy below expected threshold, check implementation")
        
    predictions = nn.predict(X_test)
    print("\\nClassification Report:")
    print(classification_report(Y_test, predictions, digits=4))
    
    # === SAVE RESULTS ===
    print("Saving results to results/metrics.json...")
    os.makedirs("results", exist_ok=True)
    metrics = {
        "loss_history": nn.loss_history,
        "val_loss_history": nn.val_loss_history,
        "accuracy_history": nn.accuracy_history,
        "val_accuracy_history": nn.val_accuracy_history
    }
    
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Saving model weights to results/model_weights.npz...")
    nn.save_weights("results/model_weights.npz")
        
    print("Done!")
