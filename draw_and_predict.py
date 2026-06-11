import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.optimizers import SGDMomentum
from src.network import NeuralNetwork
from src.layers import DenseLayer

class DrawDigitApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw a Digit (0-9)")
        
        # Build an untrained network structure
        optimizer = SGDMomentum()
        self.nn = NeuralNetwork(optimizer)
        self.nn.add_layer(DenseLayer(n_inputs=784, n_neurons=128, activation='relu'))
        self.nn.add_layer(DenseLayer(n_inputs=128, n_neurons=64, activation='relu'))
        self.nn.add_layer(DenseLayer(n_inputs=64, n_neurons=10, activation='softmax'))
        
        # Load the weights
        weights_path = "results/model_weights.npz"
        if not os.path.exists(weights_path):
            print(f"Error: {weights_path} not found!")
            print("Please run `python train.py` first to save the model weights.")
            sys.exit(1)
            
        self.nn.load_weights(weights_path)
        print("Loaded model weights successfully.")

        self.canvas_width = 280
        self.canvas_height = 280
        
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white', cursor='cross')
        self.canvas.pack(pady=10)
        
        self.old_x = None
        self.old_y = None
        
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        
        # Create a blank PIL image for prediction drawing
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        self.btn_predict = Button(self.root, text="Predict", command=self.predict_digit, font=('Arial', 14))
        self.btn_predict.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.btn_clear = Button(self.root, text="Clear", command=self.clear_canvas, font=('Arial', 14))
        self.btn_clear.pack(side=tk.RIGHT, padx=20, pady=10)
        
        self.label_result = Label(self.root, text="Prediction: None", font=('Arial', 16, 'bold'))
        self.label_result.pack(pady=10)
        
        self.root.mainloop()

    def start_stroke(self, event):
        self.old_x = event.x
        self.old_y = event.y

    def end_stroke(self, event):
        self.old_x = None
        self.old_y = None

    def paint(self, event):
        brush_width = 20
        if self.old_x and self.old_y:
            # Draw on tkinter canvas
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                    width=brush_width, fill="black", 
                                    capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
            # Draw on PIL image
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                           fill="black", width=brush_width, joint="curve")
        self.old_x, self.old_y = event.x, event.y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Prediction: None")

    def predict_digit(self):
        # 1. Resize to 28x28 using Lanczos for high-quality downsampling
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
            
        img_resized = self.image.resize((28, 28), resample_filter)
        
        # 2. Invert colors (MNIST is white stroke on black background)
        img_inverted = ImageOps.invert(img_resized)
        
        # 3. Convert to numpy array
        pixel_data = np.array(img_inverted, dtype=np.float32)
        
        # 4. Normalize to [0, 1]
        pixel_data /= 255.0
        
        # 5. Flatten and reshape to (784, 1) to match network batch dimension
        X = pixel_data.flatten().reshape(784, 1)
        
        # Pass through the network
        predictions = self.nn.predict(X)
        predicted_digit = predictions[0]
        
        self.label_result.config(text=f"Prediction: {predicted_digit}")
        print(f"Predicted: {predicted_digit}")

if __name__ == "__main__":
    app = DrawDigitApp()
