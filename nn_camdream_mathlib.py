import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def generate_matrix(sizex=4, sizey=4):
    # Erzeugt eine zufällige Matrix (nur für Testdaten)
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

def visualize_forward_pass_3d(nn, x, target=None, info=""):
    """
    Erstellt einen 3D-Graphen, der den Forward Pass visualisiert.
    X-Achse: Schichten (0: Input, 1: Hidden, 2: Output)
    Y-Achse: Neuron-Index in der jeweiligen Schicht
    Z-Achse: Aktivierungswert (bei Input: der Inputwert)
    Zusätzlich wird eine Textbox mit Input, Target und Prediction eingeblendet.
    """
    nn.forward(x)
    input_layer = [f"I {i}" for i in range(len(x))]
    hidden_layer = [f"H {i}" for i in range(len(nn.bias_hidden))]
    output_layer = [f"O {i}" for i in range(len(nn.bias_output))]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Koordinatenberechnung
    input_coords = [(0, i, x[i]) for i in range(len(x))]
    hidden_coords = [(1, i, nn.hidden_output[i]) for i in range(len(nn.bias_hidden))]
    output_coords = [(2, i, nn.output[i]) for i in range(len(nn.bias_output))]
    
    # Zeichne Input-Knoten
    for (layer, idx, val) in input_coords:
        ax.scatter(layer, idx, val, color='blue', s=100)
        ax.text(layer, idx, val, f"{input_layer[idx]}\n{val:.2f}", color='black')
    
    # Zeichne Hidden-Knoten
    for (layer, idx, act) in hidden_coords:
        z_val = nn.hidden_input[idx]
        ax.scatter(layer, idx, act, color='green', s=100)
        ax.text(layer, idx, act, f"{hidden_layer[idx]}\nz={z_val:.2f}\nσ={act:.2f}", color='black')
    
    # Zeichne Output-Knoten
    for (layer, idx, act) in output_coords:
        z_val = nn.output_input[idx]
        ax.scatter(layer, idx, act, color='red', s=100)
        ax.text(layer, idx, act, f"{output_layer[idx]}\nz={z_val:.2f}\nσ={act:.2f}", color='black')
    
    # Verbindungen Input -> Hidden
    for i, (x1, y1, z1) in enumerate(input_coords):
        for j, (x2, y2, z2) in enumerate(hidden_coords):
            weight = nn.weights_input_hidden[i, j]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='gray', linestyle='--', linewidth=0.5)
            mid = ((x1+x2)/2, (y1+y2)/2, (z1+z2)/2)
            ax.text(mid[0], mid[1], mid[2], f"{weight:.2f}", color='purple', fontsize=8)
    
    # Verbindungen Hidden -> Output
    for i, (x1, y1, z1) in enumerate(hidden_coords):
        for j, (x2, y2, z2) in enumerate(output_coords):
            weight = nn.weights_hidden_output[i, j]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='gray', linestyle='--', linewidth=0.5)
            mid = ((x1+x2)/2, (y1+y2)/2, (z1+z2)/2)
            ax.text(mid[0], mid[1], mid[2], f"{weight:.2f}", color='purple', fontsize=8)
    
    ax.set_xlabel("Layer (0: Input, 1: Hidden, 2: Output)")
    ax.set_ylabel("Neuron Index")
    ax.set_zlabel("Activation / Wert")
    ax.set_title(f"3D Forward Pass Visualization\n{info}")
    
    # Textbox mit Input, Target und Prediction
    if target is not None:
        info_text = f"Input: {np.array2string(x, precision=2)}\nTarget: {np.array2string(target, precision=2)}\nPrediction: {np.array2string(nn.output, precision=2)}"
    else:
        info_text = f"Input: {np.array2string(x, precision=2)}\nPrediction: {np.array2string(nn.output, precision=2)}"
    ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=7, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.show()

def dream(matrix_cols, matrix_rows, learning_rate, hidden_size, output_size, update_interval=2):
    """
    Nutzt die Kamera, um den Live-Feed einzulesen, downscaled das Bild auf (matrix_cols x matrix_rows),
    normalisiert es, und verwendet es als Input für das neuronale Netz. Das Netz wird fortlaufend trainiert,
    und in regelmäßigen Abständen wird der aktuelle Forward Pass in 3D visualisiert – so entsteht eine "surreale Traumwelt".
    """
    # Öffne die Kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden.")
        return
    
    # Wir erwarten, dass die Input-Größe = matrix_cols * matrix_rows entspricht.
    input_size = matrix_cols * matrix_rows
    nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, learning_rate=learning_rate)
    
    last_update = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Konvertiere zu Graustufen und downscale auf (matrix_cols x matrix_rows)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (matrix_cols, matrix_rows))
        input_matrix = small.astype(np.float32) / 255.0
        input_vector = input_matrix.flatten()  # Länge = matrix_cols * matrix_rows
        
        # Zielwert (Target) kann z.B. zufällig gewählt werden oder aus einem anderen Modul stammen
        target = np.random.rand(output_size)
        
        # Training durchführen
        nn.train(input_vector, target)
        
        # Aktualisiere Visualisierung alle update_interval Sekunden
        if time.time() - last_update > update_interval:
            prediction = nn.forward(input_vector)
            print("Prediction:", prediction)
            info = f"LR: {learning_rate}"
            visualize_forward_pass_3d(nn, input_vector, target=target, info=info)
            last_update = time.time()
        
        # Zeige den Kamerafeed
        cv2.imshow("Camera Feed - Surreal Dream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dream(matrix_cols=16, matrix_rows=16, learning_rate=0.5, hidden_size=64, output_size=16, update_interval=5)