<p align="center">
  <img src="https://raw.githubusercontent.com/nyosef1108/annpy-neopydev5454/main/assets/ANNPY%20logo2.png" width="1000" alt="Annpy Logo">
</p>

# Annpy 🧠

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/nyosef1108/annpy-neopydev5454)
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
* **⚡ High Computational Efficiency:** By leveraging optimized NumPy vectorization, **Annpy** executes complex matrix operations with remarkable speed, proving that clean code can be incredibly fast.
* **🧠 From-Scratch Implementation:** No heavy dependencies like PyTorch or TensorFlow. Just pure Python and NumPy logic.
* **📊 Built-in Preprocessing:** Features automated data normalization (scaling to [0, 1]) and shuffling to ensure your model trains effectively out of the box.
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

## 🐣 Beginner's Guide: How it Works

If you are new to Neural Networks, here is a simple breakdown of how **Annpy** processes data, using the classic **Iris Flower** dataset:

### 1. The Goal
Annpy is a mathematical tool designed to **classify** data into two distinct categories.
* **The Species:** Distinguishing between **⚪ Setosa (0)** and **⚫ Versicolor (1)**.
* **The Logic:** Let's assign a color/symbol to each classification for clarity:
  * **⚪ White:** Category **0** (Setosa)
  * **⚫ Black:** Category **1** (Versicolor)

### 2. The 4 Input Features (The Measurements)
To identify a flower, we provide the network with **4 specific measurements**. Each measurement is a "Feature" with its own color:

1. 🟢 **Sepal Length:** The length of the outer green leaves.
2. 🔵 **Sepal Width:** The width of the outer green leaves.
3. 🔴 **Petal Length:** The length of the inner colorful petals.
4. 🟡 **Petal Width:** The width of the inner colorful petals.


### 3. The Learning Phase (Training)
The network learns the "mathematical fingerprint" of each species by looking at hundreds of labeled examples:

**Example of Labeled Data:**
* **Sample A (Setosa):** `[🟢5.1, 🔵3.5, 🔴1.4, 🟡0.2]` → **Label: ⚪ 0**
* **Sample B (Versicolor):** `[🟢7.0, 🔵3.2, 🔴4.7, 🟡1.4]` → **Label: ⚫ 1**

### 📋 Dataset Structure (5-Sample List)

To help the model learn, we provide the data in two matching lists. Each item in the **Features** list corresponds directly to the same item in the **Labels** list.

The dataset is split into two synchronized arrays: X (The characteristics we measure) and y (The species we want to predict).

**The Features List (`X`)**
* **Format:** `[🟢 Sepal L, 🔵 Sepal W, 🔴 Petal L, 🟡 Petal W]`
* **Sample 1:** `[🟢5.1, 🔵3.5, 🔴1.4, 🟡0.2]`
* **Sample 2:** `[🟢4.9, 🔵3.0, 🔴1.4, 🟡0.2]`
* **Sample 3:** `[🟢7.0, 🔵3.2, 🔴4.7, 🟡1.4]`
* **Sample 4:** `[🟢6.4, 🔵3.2, 🔴4.5, 🟡1.5]`
* **Sample 5:** `[🟢5.8, 🔵2.7, 🔴4.1, 🟡1.0]`

**The Labels List (`y`)**
* **Format:** `[Species ID]`
* **Sample 1:** `⚪ 0`
* **Sample 2:** `⚪ 0`
* **Sample 3:** `⚫ 1`
* **Sample 4:** `⚫ 1`
* **Sample 5:** `⚫ 1`

### 4. The Prediction Phase
Once trained, you give the network 4 new measurements. It processes these values and produces a single output between **0** and **1**.

### 5. Understanding the Results
The result represents the network's classification. The closer to **0 (White)** or **1 (Black)**, the higher the confidence:

* **High Confidence (Setosa):** `PREDICT([🟢5.0, 🔵3.6, 🔴1.4, 🟡0.2])` → **Result: `0.02`** (Very sure it is **⚪ Setosa**)
* **High Confidence (Versicolor):** `PREDICT([🟢6.7, 🔵3.1, 🔴5.6, 🟡2.4])` → **Result: `0.98`** (Very sure it is **⚫ Versicolor**)

**What does "Uncertainty" look like?**
When features are "blurry" or sit right on the edge between species:
* **Uncertain (Leans Setosa):** `PREDICT([🟢5.5, 🔵2.5, 🔴3.0, 🟡1.1])` → **Result: `0.42`** (Slightly favors **⚪ Setosa**)
* **Uncertain (Leans Versicolor):** `PREDICT([🟢6.0, 🔵2.8, 🔴4.2, 🟡1.3])` → **Result: `0.58`** (Slightly favors **⚫ Versicolor**)

---
## 📐 Mathematical Background & Principles

### 1. Hidden Layer Transformation
The value of each neuron $j$ in the hidden layer is calculated as follows:

$$h_j = f \left( \bigl( \sum_{i=1}^{n} x_i \cdot w_{i,j} \bigr) + b_j \right)$$

**Definitions:**
*   $h_j$: The final activation (output) of the $j$-th neuron in the hidden layer.
*   $i$: Represents the specific input feature.
*   $n$: Total number of input features ($1 \leq i \leq n$).
*   $x_i$: The value of feature $i$ in the current input.
*   $w_{i,j}$: The weight connecting feature $i$ to neuron $j$.
*   $b_j$: The bias of neuron $j$.
*   $k$: Total number of neurons in the hidden layer ($1 \leq j \leq k$).
*   $f$: The Activation Function (Sigmoid).

### 2. Output Layer Calculation ($y_{pred}$)
After calculating the hidden layer activations, the final prediction of the network is computed:

$$y_{pred} = f \left( \bigl( \sum_{j=1}^{k} h_j \cdot w_{j,out} \bigr) + b_{out} \right)$$

**Definitions:**
*   $y_{pred}$: The final prediction of the network (between 0 and 1).
*   $h_j$: The activation value from the $j$-th neuron in the hidden layer.
*   $w_{j,out}$: The weight connecting hidden neuron $j$ to the output neuron.
*   $b_{out}$: The bias of the output neuron.
*   $k$: Total number of neurons in the hidden layer ($1 \leq j \leq k$).

### 3. Measuring Error: Mean Squared Error (MSE)
To evaluate the network's performance over a set of $m$ samples, we calculate the average squared difference between the true labels and the predictions:

$$Loss = \frac{1}{m} \sum_{i=1}^{m} (y_{true}^{(i)} - y_{pred}^{(i)})^2$$

**Definitions:**
*   $m$: The total number of samples in the dataset (or batch).
*   $\sum_{i=1}^{m}$: The sum of errors across all samples from $1$ to $m$.
*   $y_{true}^{(i)}$: The actual ground truth for sample $i$.
*   $y_{pred}^{(i)}$: The network's prediction for sample $i$.
*   $Loss$: The final average error used to guide the optimization process.

### 4. Gradient Descent & Weight Update
To minimize the $Loss$, we adjust each weight and bias in the direction that reduces the error using the **Chain Rule**.

#### A. The Update Rule:
We update all weights and biases in the network (both hidden and output layers) using the calculated gradients:

**1. Hidden Layer Parameters ($w_{i,j}$ and $b_j$):**

$$w_{i,j} = w_{i,j} - \eta \cdot \frac{\partial Loss}{\partial w_{i,j}}$$


$$b_j = b_j - \eta \cdot \frac{\partial Loss}{\partial b_j}$$


**2. Output Layer Parameters ($w_{j,out}$ and $b_{out}$):**

$$w_{j,out} = w_{j,out} - \eta \cdot \frac{\partial Loss}{\partial w_{j,out}}$$


$$b_{out} = b_{out} - \eta \cdot \frac{\partial Loss}{\partial b_{out}}$$

#### B. The Derivative (Chain Rule):

To calculate the gradient for each parameter, we break the influence down into the specific components along the path to the final error:


**1. Hidden Layer Weight ($\frac{\partial Loss}{\partial w_{i,j}}$):**

$$\frac{\partial Loss}{\partial w_{i,j}} = \frac{\partial Loss}{\partial y_{pred}} \cdot \frac{\partial y_{pred}}{\partial h_j} \cdot \frac{\partial h_j}{\partial w_{i,j}}$$


**2. Hidden Layer Bias ($\frac{\partial Loss}{\partial b_j}$):**

$$\frac{\partial Loss}{\partial b_j} = \frac{\partial Loss}{\partial y_{pred}} \cdot \frac{\partial y_{pred}}{\partial h_j} \cdot \frac{\partial h_j}{\partial b_j}$$


**3. Output Layer Weight ($\frac{\partial Loss}{\partial w_{j,out}}$):**

$$\frac{\partial Loss}{\partial w_{j,out}} = \frac{\partial Loss}{\partial y_{pred}} \cdot \frac{\partial y_{pred}}{\partial w_{j,out}}$$


**4. Output Layer Bias ($\frac{\partial Loss}{\partial b_{out}}$):**

$$\frac{\partial Loss}{\partial b_{out}} = \frac{\partial Loss}{\partial y_{pred}} \cdot \frac{\partial y_{pred}}{\partial b_{out}}$$


**Definitions:**
* **$\eta$ (Eta):** The **Learning Rate** – a small constant (e.g., 0.1) that determines the step size toward the minimum.

#### C. Mathematical Breakdown of the Gradient:

By applying the derivatives to each component of our network, we get:


**The Error Gradient:**

$$\frac{\partial Loss}{\partial y_{pred}} = \frac{\partial \left( y_{true} - y_{pred} \right)^2}{\partial y_{pred}} = -2 \cdot \left( y_{true} - y_{pred} \right)$$


**The Output Activation (with respect to hidden neurons):**

$$\frac{\partial y_{pred}}{\partial h_j} = \frac{\partial f \left( \left( \sum_{j=1}^{k} h_j \cdot w_{j,out} \right) + b_{out} \right)}{\partial h_j} = f' \left( \left( \sum_{j=1}^{k} h_j \cdot w_{j,out} \right) + b_{out} \right) \cdot w_{j,out}$$


**The Hidden Weight Impact:**

$$\frac{\partial h_j}{\partial w_{i,j}} = \frac{\partial f \left( \left( \sum_{i=1}^{n} x_i \cdot w_{i,j} \right) + b_j \right)}{\partial w_{i,j}} = f' \left( \left( \sum_{i=1}^{n} x_i \cdot w_{i,j} \right) + b_j \right) \cdot x_i$$


**The Hidden Bias Impact:**

$$\frac{\partial h_j}{\partial b_j} = \frac{\partial f \left( \left( \sum_{i=1}^{n} x_i \cdot w_{i,j} \right) + b_j \right)}{\partial b_j} = f' \left( \left( \sum_{i=1}^{n} x_i \cdot w_{i,j} \right) + b_j \right) \cdot 1$$


**The Output Weight Impact:**

$$\frac{\partial y_{pred}}{\partial w_{j,out}} = \frac{\partial f \left( \left( \sum_{j=1}^{k} h_j \cdot w_{j,out} \right) + b_{out} \right)}{\partial w_{j,out}} = f' \left( \left( \sum_{j=1}^{k} h_j \cdot w_{j,out} \right) + b_{out} \right) \cdot h_j$$


**The Output Bias Impact:**

$$\frac{\partial y_{pred}}{\partial b_{out}} = \frac{\partial f \left( \left( \sum_{j=1}^{k} h_j \cdot w_{j,out} \right) + b_{out} \right)}{\partial b_{out}} = f' \left( \left( \sum_{j=1}^{k} h_j \cdot w_{j,out} \right) + b_{out} \right) \cdot 1$$
#### D. The Sigmoid Derivative ($f'$)
To implement the backpropagation, we need the derivative of our activation function. The Sigmoid function $f(x) = \frac{1}{1 + e^{-x}}$ has a unique and convenient derivative:


$$f'(x) = f(x) \cdot (1 - f(x))$$


**Why this is useful:**
Since we already calculated $f(x)$ (the neuron's output) during the **Forward Pass**, calculating the gradient during the **Backward Pass** is computationally efficient—we simply reuse the existing output.

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

