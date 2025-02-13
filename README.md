# img2shape

## Overview

**img2shape** is an experimental project that explores the intersection of artificial intelligence, neural networks, and image recognition. The project focuses on:
- **Shape Detection:** Identifying and highlighting shapes within images.
- **Motion Detection:** Recognizing faces, hands, and other forms in videos.
- **Neural Network Visualization:** Configuring and visualizing neural network structures and behaviors.

## File Overview

- **`img2shape.py`**  
  Detects shapes in static images and fills them with color. Ideal for experimenting with artistic form recognition.

- **`mov2shape.py`**  
  Processes video input to detect faces, hands, and various shapes in motion, extending the functionality to real-time or recorded video streams.

- **`nn_xor_graph.py`**  
  Visualizes a configurable neural network. You can modify the input matrix, hidden layers, and see both forward and backward predictions. 

- **`nn_xor.py`**  
  Implements a simple neural network with a 4x4 input matrix.

## Installation

### Prerequisites

- **Python 3.8+** (recommended)
- Common Python libraries for image processing and numerical computation.

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://your-repo-url.git
   cd img2shape
   ```

2. **Create a virtual environment (optional)**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

## Usage
    python img2shape.py --input path/to/image.jpg
    python mov2shape.py --input path/to/video.mp4
    python nn_xor_graph.py

### Contributing    
Contributions are welcome! If you have suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

### License
This project is released under the MIT License.

This README provides a clear overview of the project, detailed information about each file, installation steps, configuration options, and usage instructions. Enjoy exploring and expanding **img2shape**!