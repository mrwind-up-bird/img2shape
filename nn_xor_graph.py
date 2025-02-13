import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import tkinter as tk  

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def generate_matrix(sizex=4, sizey=4):
    matrix = [[random.randint(0, 1) for _ in range(sizex)] for _ in range(sizey)]
    print(f"Matrix: {matrix}")
    return matrix

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.learning_rate = learning_rate        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.random.randn(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.random.randn(output_size)
    
    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output
    
    def backward(self, x, target):
        
        output_error = target - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.learning_rate * np.outer(self.hidden_output, output_delta)
        self.bias_output += self.learning_rate * output_delta        
        self.weights_input_hidden += self.learning_rate * np.outer(x, hidden_delta)
        self.bias_hidden += self.learning_rate * hidden_delta
    
    def train(self, x, target):
        self.forward(x)
        self.backward(x, target)

def visualize_forward_pass_3d(nn, x, target=None, epoch_info=""):

    nn.forward(x)
    
    input_layer = [f"I {i}" for i in range(len(x))]
    hidden_layer = [f"H {i}" for i in range(len(nn.bias_hidden))]
    output_layer = [f"O {i}" for i in range(len(nn.bias_output))]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Input: x=0, y = Index, z = Input-Wert
    input_coords = [(0, i, x[i]) for i in range(len(x))]
    # Hidden: x=1, y = Index, z = threshold (σ(z)); additional z = hidden_input
    hidden_coords = [(1, i, nn.hidden_output[i]) for i in range(len(nn.bias_hidden))]
    # Output: x=2, y = Index, z = threshold (σ(z)); additional z = output_input
    output_coords = [(2, i, nn.output[i]) for i in range(len(nn.bias_output))]
    
    for (layer, idx, act) in input_coords:
        ax.scatter(layer, idx, act, color='blue', s=100)
        ax.text(layer, idx, act, f"{input_layer[idx]}\n{act:.2f}", color='black')
    
    for (layer, idx, act) in hidden_coords:
        z_val = nn.hidden_input[idx]
        ax.scatter(layer, idx, act, color='green', s=100)
        ax.text(layer, idx, act, f"{hidden_layer[idx]}\nz={z_val:.2f}\nσ={act:.2f}", color='black')
    
    for (layer, idx, act) in output_coords:
        z_val = nn.output_input[idx]
        ax.scatter(layer, idx, act, color='red', s=100)
        ax.text(layer, idx, act, f"{output_layer[idx]}\nz={z_val:.2f}\nσ={act:.2f}", color='black')
    
    for i, (x1, y1, z1) in enumerate(input_coords):
        for j, (x2, y2, z2) in enumerate(hidden_coords):
            weight = nn.weights_input_hidden[i, j]
            xs = [x1, x2]
            ys = [y1, y2]
            zs = [z1, z2]
            ax.plot(xs, ys, zs, color='gray', linestyle='--', linewidth=0.5)
            mid_x, mid_y, mid_z = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
            ax.text(mid_x, mid_y, mid_z, f"{weight:.2f}", color='purple', fontsize=8)
    
    for i, (x1, y1, z1) in enumerate(hidden_coords):
        for j, (x2, y2, z2) in enumerate(output_coords):
            weight = nn.weights_hidden_output[i, j]
            xs = [x1, x2]
            ys = [y1, y2]
            zs = [z1, z2]
            ax.plot(xs, ys, zs, color='gray', linestyle='--', linewidth=0.5)
            mid_x, mid_y, mid_z = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
            ax.text(mid_x, mid_y, mid_z, f"{weight:.2f}", color='purple', fontsize=8)
    
    ax.set_xlabel("Layer (0: Input, 1: Hidden, 2: Output)")
    ax.set_ylabel("Neuron Index")
    ax.set_zlabel("Activation / Wert")
    ax.set_title(f"3D Forward Pass Visualization\n{epoch_info}")
    
    if target is not None:
        info_text = f"Input: {np.array2string(x, precision=2)}\nTarget: {np.array2string(target, precision=2)}\nPrediction: {np.array2string(nn.output, precision=2)}"
    else:
        info_text = f"Input: {np.array2string(x, precision=2)}\nPrediction: {np.array2string(nn.output, precision=2)}"
    ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.show()

def run_experiment(matrix_cols, matrix_rows, learning_rate, hidden_size, output_size, epochs):
    X = np.array(generate_matrix(matrix_cols, matrix_rows))
    y = np.array(generate_matrix(output_size, matrix_rows))
    
    nn = NeuralNetwork(input_size=matrix_cols, hidden_size=hidden_size, output_size=output_size, learning_rate=learning_rate)
    
    for epoch in range(epochs):
        for i in range(len(X)):
            nn.train(X[i], y[i])
        if epoch % (epochs // 10) == 0:
            outputs = np.array([nn.forward(x) for x in X])
            loss = np.mean((y - outputs) ** 2)
            print(f"Epoch {epoch}, Loss: {loss}")
    
    test_input = X[0]
    test_target = y[0]
    prediction = nn.forward(test_input)
    print("Test input:", test_input)
    print("Target:", test_target)
    print("Prediction:", prediction)
    visualize_forward_pass_3d(nn, test_input, target=test_target, epoch_info=f"Epochs: {epochs}, LR: {learning_rate}")

def main_gui():
    root = tk.Tk()
    root.title("Neural Network Parameter Tuner")
    
    params = {
        "Matrix Columns": 16,
        "Matrix Rows": 16,
        "Learning Rate": 0.2,
        "Hidden Size": 8,
        "Output Size": 2,
        "Epochs": 50000
    }
    entries = {}
    row = 0
    for key, val in params.items():
        tk.Label(root, text=key).grid(row=row, column=0, padx=5, pady=5, sticky="w")
        entry = tk.Entry(root)
        entry.insert(0, str(val))
        entry.grid(row=row, column=1, padx=5, pady=5)
        entries[key] = entry
        row += 1
    
    def on_run():
        matrix_cols = int(entries["Matrix Columns"].get())
        matrix_rows = int(entries["Matrix Rows"].get())
        lr = float(entries["Learning Rate"].get())
        hidden = int(entries["Hidden Size"].get())
        output = int(entries["Output Size"].get())
        epochs = int(entries["Epochs"].get())
        run_experiment(matrix_cols, matrix_rows, lr, hidden, output, epochs)
    
    run_button = tk.Button(root, text="Train and Visualize", command=on_run)
    run_button.grid(row=row, column=0, columnspan=2, padx=5, pady=10)
    
    explanation = ("(C)2025 mrwind-up-bird - oliver.baer@gmail.com \n"
                   "https://github.com/mrwind-up-bird/img2shape")
    tk.Label(root, text=explanation, justify="left").grid(row=row+1, column=0, columnspan=2, padx=5, pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    main_gui()