import cv2
import numpy as np
import random
import math
import time
import os

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.learning_rate = learning_rate        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.random.randn(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.random.randn(output_size)
    
    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)
        return self.output
    
    def backward(self, x, target):
        output_error = target - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += self.learning_rate * np.outer(self.hidden_output, output_delta)
        self.bias_output += self.learning_rate * output_delta        
        self.weights_input_hidden += self.learning_rate * np.outer(x, hidden_delta)
        self.bias_hidden += self.learning_rate * hidden_delta
    
    def train(self, x, target):
        self.forward(x)
        self.backward(x, target)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

class OverlayGenerator:
    def __init__(self, output_size, grid_cols=None, colormap=cv2.COLORMAP_JET):
        self.output_size = output_size
        self.colormap = colormap
        if grid_cols is None:
            sqrt_n = math.sqrt(output_size)
            self.grid_cols = int(sqrt_n) if sqrt_n.is_integer() else int(np.ceil(sqrt_n))
        else:
            self.grid_cols = grid_cols
        self.grid_rows = int(np.ceil(output_size / self.grid_cols))
    
    def generate_overlay(self, nn_output, width, height):
        overlay_small = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        for i in range(self.output_size):
            row = i // self.grid_cols
            col = i % self.grid_cols
            intensity = int(nn_output[i] * 255)
            overlay_small[row, col] = intensity
        
        # colormap
        overlay_color = cv2.applyColorMap(overlay_small, self.colormap)
        overlay_large = cv2.resize(overlay_color, (width, height), interpolation=cv2.INTER_NEAREST)
        return overlay_large

class FaceDetector:
    def __init__(self, cascade_path="cascades/haarcascade_frontalface_default.xml"):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise Exception("Error: cant find face cascade")
    
    def detect_face(self, gray_frame):
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            return faces[0] # returns first face found
        return None

def create_tracker():
    tracker = None
    try:
        tracker = cv2.TrackerCSRT_create()
        return tracker
    except Exception:
        pass
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
        return tracker
    except Exception:
        pass
    try:
        tracker = cv2.TrackerKCF_create()
        return tracker
    except Exception:
        pass
    try:
        tracker = cv2.legacy.TrackerKCF_create()
        return tracker
    except Exception:
        pass
    try:
        tracker = cv2.TrackerMOSSE_create()
        return tracker
    except Exception:
        pass
    try:
        tracker = cv2.legacy.TrackerMOSSE_create()
        return tracker
    except Exception:
        pass
    raise Exception("tracker (CSRT, KCF oder MOSSE) not found.")

class CameraDream:
    def __init__(self, nn, overlay_generator, matrix_cols, matrix_rows, face_detector, training_iterations=100):
        self.nn = nn
        self.overlay_generator = overlay_generator
        self.matrix_cols = matrix_cols
        self.matrix_rows = matrix_rows
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Kamera konnte nicht geöffnet werden.")
        self.face_detector = face_detector
        self.tracker = None  # initialise if face found
        self.training_iterations = training_iterations  # training count
    
    def train_on_face(self, roi_gray):
        roi_resized = cv2.resize(roi_gray, (self.matrix_cols, self.matrix_rows))
        input_vector = roi_resized.astype(np.float32) / 255.0
        input_vector = input_vector.flatten()
       
        # targetmatrix
        target = self.create_gradient()
        
        for _ in range(self.training_iterations):
            self.nn.train(input_vector, target)

        np.savez("model.npz", 
                 weights_input_hidden=self.nn.weights_input_hidden, 
                 bias_hidden=self.nn.bias_hidden, 
                 weights_hidden_output=self.nn.weights_hidden_output, 
                 bias_output=self.nn.bias_output)
        print("finished training:", target)
        
    # smilie matrix
    def create_matrix(self):
        matrix = np.array([
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
        return matrix.flatten()
    
    # gradient
    def create_gradient(self):
        output_size = 32
        return np.linspace(0, 1, output_size)
    
    # random
    def create_random_matrix(self):
        return np.random.rand(self.nn.weights_hidden_output.shape[1])
    
    def run(self):
        trained = False
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_height, frame_width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # initialise tracker
            if self.tracker is None:
                face = self.face_detector.detect_face(gray)
                if face is not None:
                    (x, y, w, h) = face
                    # create tracker
                    self.tracker = create_tracker()
                    self.tracker.init(frame, tuple(face))
                    
                    # initialise training, otherwise use pre-trained model
                    if os.path.exists("model.npz"):
                        data = np.load("model.npz")
                        self.nn.weights_input_hidden = data["weights_input_hidden"]
                        self.nn.bias_hidden = data["bias_hidden"]
                        self.nn.weights_hidden_output = data["weights_hidden_output"]
                        self.nn.bias_output = data["bias_output"]
                        print (f"load trainingsmodel")
                    else:
                        roi = gray[y:y+h, x:x+w]
                        self.train_on_face(roi)
            else:
                ok, bbox = self.tracker.update(frame)
                if ok:
                    (x, y, w, h) = [int(v) for v in bbox]
                else:
                    # lost tracker, return
                    self.tracker = None
                    x, y, w, h = 0, 0, frame_width, frame_height
            
            # choose face as input vector
            if self.tracker is not None:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi = gray[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (self.matrix_cols, self.matrix_rows))
                input_vector = roi_resized.astype(np.float32) / 255.0
                input_vector = input_vector.flatten()
            else:
                small = cv2.resize(gray, (self.matrix_cols, self.matrix_rows))
                input_vector = small.astype(np.float32) / 255.0
                input_vector = input_vector.flatten()
            
            # nn forward pass / output neurons
            nn_output = self.nn.forward(input_vector)
            if self.tracker is not None:
                overlay = self.overlay_generator.generate_overlay(nn_output, w, h)
                roi_color = frame[y:y+h, x:x+w]
                blended_roi = cv2.addWeighted(roi_color, 0.5, overlay, 0.5, 0)
                frame[y:y+h, x:x+w] = blended_roi
            else:
                overlay = self.overlay_generator.generate_overlay(nn_output, frame_width, frame_height)
                frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
            
            cv2.imshow("Lax Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    # downscaling the neuronal input 
    matrix_cols = 16
    matrix_rows = 16
    input_size = matrix_cols * matrix_rows
    
    # define output and hidden neuron count
    hidden_size = 64
    output_size = 16
    learning_rate = 0.2
    
    nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, learning_rate=learning_rate)
    # colormap
    overlay_gen = OverlayGenerator(output_size=output_size, colormap=cv2.COLORMAP_JET)
    face_detector = FaceDetector()
    # training iterations
    camera_dream = CameraDream(nn, overlay_gen, matrix_cols, matrix_rows, face_detector, training_iterations=100)
    
    camera_dream.run()

if __name__ == "__main__":
    main()