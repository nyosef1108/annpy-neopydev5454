# Annpy 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/dependency-NumPy-blueviolet.svg)](https://numpy.org/)

**Annpy** is a lightweight, efficient, and fully transparent Artificial Neural Network (ANN) implementation built from scratch using NumPy. 

While many frameworks are "black boxes," **Annpy** is designed to be customizable and clear, making it perfect for educational purposes, lightweight integrations, and fast prototyping.

---

## ✨ Key Features

* **🎯 Total Architectural Flexibility:**
    * **Any Input Size:** Configure the network to accept any number of input features—from 2 to 2,000+.
    * **Customizable Hidden Layer:** You define the number of neurons in the hidden layer to perfectly balance learning power and processing speed.
* **🧠 From-Scratch Implementation:** No heavy dependencies like PyTorch or TensorFlow. Just pure Python and NumPy logic.
* **⚡ Built-in Preprocessing:** Features automated data normalization (scaling to [0, 1]) and shuffling to ensure your model trains effectively out of the box.
* **🛠️ Developer Friendly:** Clean, documented code that is easy to read, extend, and debug.

---

## 🚀 Installation

Install **Annpy** directly via pip:

```bash
pip install annpy-neopydev5454

```

---

## 🛠️ Quick Start: The Iris Dataset Example

In this example, we configure Annpy to classify a dataset of 20 samples. The network is set to receive 4 inputs and uses a hidden layer of 10 neurons.

```python
import numpy as np
from annpy import ANN

# 1. Prepare synthetic data (20 samples, 4 features each)
X = np.array([
    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5],
    [5.8, 2.7, 4.1, 1.0], [5.1, 2.5, 3.0, 1.1], [6.7, 3.1, 5.6, 2.4], [6.3, 2.3, 4.4, 1.3],
    [5.0, 3.4, 1.5, 0.2], [5.9, 3.0, 5.1, 1.8], [5.2, 3.5, 1.5, 0.2], [4.7, 3.2, 1.3, 0.2],
    [6.9, 3.1, 4.9, 1.5], [5.4, 3.9, 1.7, 0.4], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
    [6.3, 3.3, 4.7, 1.6], [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [6.1, 2.8, 4.0, 1.3]
])

# 2. Prepare target labels (Must match X length)
y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1])

# 3. Initialize the model
# input_size=4 (features), hidden_layer_size=10 (neurons)
model = ANN(hidden_layer_size=10, input_size=4)

# 4. Train the model
# X: The training features matrix
# y: The target labels (must be the same length as X)
# learning_rate: The step size for weight updates
# epochs: Number of times to iterate over the entire dataset
model.train(X, y, learning_rate=0.1, epochs=1500)

# 5. Predict a new sample
# Note: Since this sample is similar to the Setosa class (class 0), the prediction should be as close as possible to 0.
new_sample = np.array([5.0, 3.6, 1.4, 0.3])
prediction = model.predict(new_sample)
print(f"Prediction: {prediction}")

```

---

## 📊 How it Works

Annpy implements a **Multi-Layer Perceptron (MLP)** with one hidden layer:

1. **Dynamic Weights:** Initialized based on your `input_size` and `hidden_layer_size`.
2. **Forward Pass:** Uses the Sigmoid activation function to compute the output.
3. **Backpropagation:** Updates weights using Gradient Descent.
4. **Auto-Normalization:** Automatically scales input data to [0, 1] for better convergence.

---

## 📜 License

Distributed under the **MIT License**. See the `LICENSE` file for more information.

---

Created with ❤️ by [Nehorai Yosef](https://github.com/nyosef1108) | [Contact](mailto:neopydev5454@gmail.com)

