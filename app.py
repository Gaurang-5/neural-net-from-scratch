import gradio as gr
from PIL import Image, ImageOps
import numpy as np
import os
import sys

# Load model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.optimizers import SGDMomentum
from src.network import NeuralNetwork
from src.layers import DenseLayer

# Re-initialize the exact same architecture
optimizer = SGDMomentum()
nn = NeuralNetwork(optimizer)
nn.add_layer(DenseLayer(n_inputs=784, n_neurons=128, activation='relu'))
nn.add_layer(DenseLayer(n_inputs=128, n_neurons=64, activation='relu'))
nn.add_layer(DenseLayer(n_inputs=64, n_neurons=10, activation='softmax'))

# Path where huggingface or local environment will find the weights
weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "model_weights.npz")
if os.path.exists(weights_path):
    nn.load_weights(weights_path)
else:
    print("Warning: model_weights.npz not found. Using untrained weights.")

def predict_digit(image_dict):
    if image_dict is None:
        return {str(i): 0.0 for i in range(10)}
    
    # Gradio 4+ Sketchpad returns a dictionary with a 'composite' layer
    # Fallback to direct image if different Gradio version is used
    if isinstance(image_dict, dict) and "composite" in image_dict:
        img = image_dict["composite"]
    else:
        img = image_dict
        
    # Check if image is empty (all transparent)
    if not img.getbbox():
        return {str(i): 0.0 for i in range(10)}
    
    # Create a white background and paste the drawn image (which has a transparent background)
    bg = Image.new("RGBA", img.size, "white")
    bg.paste(img, (0, 0), img)
    img = bg.convert("L")  # Convert to grayscale
    
    # Resize to 28x28
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    img_resized = img.resize((28, 28), resample)
    
    # Invert colors (MNIST is white strokes on black background)
    img_inverted = ImageOps.invert(img_resized)
    
    # Normalize
    pixel_data = np.array(img_inverted, dtype=np.float32) / 255.0
    
    # Reshape for network (784, 1)
    X = pixel_data.flatten().reshape(784, 1)
    
    # Get probabilities via a manual forward pass
    probabilities = nn.forward(X).flatten()
    
    # Return a dictionary of class names to their confidence scores
    return {str(i): float(probabilities[i]) for i in range(10)}

# Build the Gradio UI
with gr.Blocks(title="Neural Network from Scratch") as demo:
    gr.Markdown(
        """
        # 🧠 Neural Network from Scratch
        Draw a digit (0-9) below and see the real-time prediction from the custom-built, pure-NumPy neural network! 
        This model was trained without any ML frameworks (no PyTorch, no TensorFlow).
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Sketchpad allows the user to draw
            sketchpad = gr.Sketchpad(label="Draw a number here", type="pil")
            clear_btn = gr.Button("Clear Canvas")
            
        with gr.Column(scale=1):
            # Label displays the confidence bars for the top predictions
            label = gr.Label(num_top_classes=3, label="Top Predictions")
            
    # Whenever the drawing changes, update the prediction automatically
    sketchpad.change(predict_digit, inputs=sketchpad, outputs=label)
    clear_btn.click(lambda: None, inputs=None, outputs=sketchpad)

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
