#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def generate_matrix(sizex=4, sizey=4):
    rng = random.Random(1)
    matrix = [[rng.randint(0,1) for _ in range(sizex)] for _ in range(sizey)]
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
        # error output neuron
        output_error = target - self.output
        
        # delta output neuron
        output_delta = output_error * sigmoid_derivative(self.output)
        
        # hidden error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # weights / biasses
        self.weights_hidden_output += self.learning_rate * np.outer(self.hidden_output, output_delta)
        self.bias_output += self.learning_rate * output_delta
        
        self.weights_input_hidden += self.learning_rate * np.outer(x, hidden_delta)
        self.bias_hidden += self.learning_rate * hidden_delta
    
    def train(self, x, target):
        self.forward(x)
        self.backward(x, target)

def main():
    
    X = np.array(generate_matrix(4,4))
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])
    
    nn = NeuralNetwork(input_size=4, hidden_size=6, output_size=4, learning_rate=0.1)
    
    epochs = 100000
    for epoch in range(epochs):
        for i in range(len(X)):
            nn.train(X[i], y[i])
        
        if epoch % 1000 == 0:
            outputs = np.array([nn.forward(x) for x in X])
            loss = np.mean((y - outputs) ** 2)
            print(f"Epoch {epoch}, Loss: {loss}")
    
    # test
    print("\nResults:")
    for x in X:
        prediction = nn.forward(x)
        print(f"Input: {x} -> Output: {prediction}")

if __name__ == "__main__":
    main()