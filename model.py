import numpy as np

class ANN:
    def __init__(self, hidden_layer_size, input_size):
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.HL_before_activation = []
        self.HL_after_activation = []
        self.OL_before_activation = 0
        self.OL_after_activation = 0
        self.HL_weights, self.OL_weights, self.HL_biases, self.OL_bias = self.initialize_weights_and_biases(hidden_layer_size, input_size)
        # column in self.HL_weights == weights of a single neuron in the hidden layer.
        self.col_min = None
        self.col_max = None

    def print_ANN(self):
        """
        Prints a detailed and formatted summary of the Neural Network architecture,
        including weights and biases for each neuron.
        """
        print("\n" + "="*60)
        print(f"{'DETAILED NEURAL NETWORK STRUCTURE':^60}")
        print("="*60)

        # Layer 1: Hidden Layer Details
        print(f"\n[ LAYER 1: HIDDEN LAYER ]")
        print(f"Neurons: {self.hidden_layer_size} | Activation: Sigmoid")
        print("-" * 30)
        
        # Iterate through each neuron in the hidden layer
        # HL_weights shape is (input_size, hidden_layer_size)
        for i in range(self.hidden_layer_size):
            # Extract weights connected to the i-th hidden neuron
            weights = self.HL_weights[:, i]  
            bias = self.HL_biases[0, i]
            weights_str = ", ".join([f"{w:.3f}" for w in weights])
            print(f" Neuron {i+1}:")
            print(f"  - Weights (Incoming): [{weights_str}]")
            print(f"  - Bias: {bias:.3f}")
        
        print("\n" + "-" * 60)

        # Layer 2: Output Layer Details
        print(f"[ LAYER 2: OUTPUT LAYER ]")
        print(f"Neurons: 1 | Activation: Sigmoid")
        print("-" * 30)
        
        # Detail for the single output neuron
        # OL_weights shape is (hidden_layer_size, 1)
        ol_weights_str = ", ".join([f"{w[0]:.3f}" for w in self.OL_weights])
        print(f" Neuron 1 (Final):")
        print(f"  - Weights (From Hidden): [{ol_weights_str}]")
        print(f"  - Bias: {self.OL_bias[0]:.3f}")

        print("="*60 + "\n")

    def initialize_weights_and_biases(self, hidden_layer_size, input_size):
        # HL_weights is a 2D array: number of rows = input_size, number of columns = hidden_layer_size
        HL_weights = np.random.rand(input_size, hidden_layer_size)
        
        # OL_weights is a 2D array: number of rows = hidden_layer_size, number of columns = 1 (output layer has 1 neuron)
        OL_weights = np.random.rand(hidden_layer_size, 1)
        
        # HL_biases is a 2D array: number of rows = hidden_layer_size, number of columns = 1 (each neuron in hidden layer has its own bias)
        HL_biases = np.random.rand(1, hidden_layer_size)
        
        # OL_bias is a scalar (a single value)
        OL_bias = np.random.rand(1)

        return HL_weights, OL_weights, HL_biases, OL_bias
            
    def forward_feeding(self, inputs): 
        self.HL_before_activation = np.dot(inputs, self.HL_weights) + self.HL_biases        
        self.HL_after_activation = self.sigmoid(self.HL_before_activation)        
        self.OL_before_activation = np.dot(self.HL_after_activation, self.OL_weights) + self.OL_bias        
        self.OL_after_activation = self.sigmoid(self.OL_before_activation)
     
    def backward_propagation(self,input, learning_rate, y_true):
        index = 0
        # number of weights in hidden layer = hidden_layer_size * input_size
        for weight in np.nditer(self.HL_weights):
            d_y_pred_d_HL_after_activation = self.sigmoid_derivative(self.OL_before_activation.flat[0]) * self.OL_weights.flat[index // self.input_size]
            d_HL_after_activation_d_W = self.sigmoid_derivative(self.HL_before_activation.flat[index // self.input_size]) * input.flat[index % self.input_size]
            d_loss_d_W = self.d_loss_d_y_pred(self.OL_after_activation.flat[0], y_true) * d_y_pred_d_HL_after_activation * d_HL_after_activation_d_W
            
            # update the weight
            np.put(self.HL_weights, index, self.HL_weights.flat[index] - learning_rate * d_loss_d_W)
            index += 1
            
        index = 0
        # number of biases in hidden layer = hidden_layer_size
        for bias in np.nditer(self.HL_biases):
            d_y_pred_d_HL_after_activation = self.sigmoid_derivative(self.OL_before_activation.flat[0]) * self.OL_weights.flat[index]
            d_HL_after_activation_d_bias = self.sigmoid_derivative(self.HL_before_activation.flat[index]) 
            d_loss_d_bias = self.d_loss_d_y_pred(self.OL_after_activation.flat[0], y_true) * d_y_pred_d_HL_after_activation * d_HL_after_activation_d_bias
            
            # update the bias
            np.put(self.HL_biases, index, self.HL_biases.flat[index] - learning_rate * d_loss_d_bias)
            index += 1
            
        index = 0
        # number of weights in output layer = hidden_layer_size 
        for weight in np.nditer(self.OL_weights):
            d_y_pred_d_weight = self.sigmoid_derivative(self.OL_before_activation.flat[0]) * self.HL_after_activation.flat[index]
            d_loss_d_weight = self.d_loss_d_y_pred(self.OL_after_activation.flat[0], y_true) * d_y_pred_d_weight
        
        # update the weight
            np.put(self.OL_weights, index, self.OL_weights.flat[index] - learning_rate * d_loss_d_weight)
            index += 1
        
        index = 0
        # number of biases in output layer = 1
        for bias in self.OL_bias:
            d_y_pred_d_bias = self.sigmoid_derivative(self.OL_before_activation.flat[0]) 
            d_loss_d_bias = self.d_loss_d_y_pred(self.OL_after_activation.flat[0], y_true) * d_y_pred_d_bias
            
            # update the bias
            np.put(self.OL_bias, index, self.OL_bias.flat[index] - learning_rate * d_loss_d_bias)
            index += 1
     
    def train(self, inputs, y_true, learning_rate, epochs, do_early_stop=False, patience=100):
        inputs = self.normalize_dataset(inputs)
        inputs, y_true = self.shuffle_data(inputs, y_true)
        
        loss_history = []
        epoch_axis = []
        
        best_loss = float('inf')
        epochs_without_improvement = 0
        best_weights = (self.HL_weights.copy(), self.OL_weights.copy())
        best_biases = (self.HL_biases.copy(), self.OL_bias.copy())

        for epoch in range(epochs):
            for input_sample, target in zip(inputs, y_true):
                self.forward_feeding(input_sample)
                self.backward_propagation(input_sample, learning_rate, target)
            
            all_preds = []
            for i in inputs:
                h = self.sigmoid(np.dot(i, self.HL_weights) + self.HL_biases)
                o = self.sigmoid(np.dot(h, self.OL_weights) + self.OL_bias)
                all_preds.append(o.flat[0])
            
            current_loss = self.loss(y_true, np.array(all_preds))
            loss_history.append(current_loss)
            epoch_axis.append(epoch)

            if current_loss < best_loss:
                best_loss = current_loss
                best_weights = (self.HL_weights.copy(), self.OL_weights.copy())
                best_biases = (self.HL_biases.copy(), self.OL_bias.copy())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if do_early_stop and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}. Restoring best weights.")
                break

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {current_loss:.6f}")

        print(f"\nTraining finished. Restoring best weights found (Best Loss: {best_loss:.6f})")
        self.HL_weights, self.OL_weights = best_weights
        self.HL_biases, self.OL_bias = best_biases

        self.print_ANN()
        
    def loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()   
    
    def d_loss_d_y_pred(self, y_pred, y_true):
        # loss = (y_true - y_pred) ** 2
        return -2 * (y_true - y_pred)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def normalize_dataset(self, data):
        data = np.array(data, dtype=float)
        self.col_min = data.min(axis=0)
        self.col_max = data.max(axis=0)
        range_val = self.col_max - self.col_min
        range_val[range_val == 0] = 1.0
        return (data - self.col_min) / range_val
        
    def shuffle_data(self, inputs, y_true):
        inputs = np.array(inputs)
        y_true = np.array(y_true)
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        return inputs[indices], y_true[indices]
    
    def print_results(self, inputs, y_true):
        norm_inputs = self.normalize_dataset(inputs)
        correct = 0
        wrong = 0
        
        for i in range(len(norm_inputs)):
            self.forward_feeding(norm_inputs[i])
            pred = 1 if self.OL_after_activation > 0.5 else 0
            
            if pred == y_true[i]:
                correct += 1
            else:
                wrong += 1
                
        print("\n--- Final Results ---")
        print(f"Total samples: {len(inputs)}")
        print(f"Correct predictions: {correct}")
        print(f"Wrong predictions: {wrong}")
        print(f"Accuracy: {(correct/len(inputs))*100:.2f}%")
        
    def predict(self, input_sample):
        input_sample = np.array(input_sample, dtype=float)
        if self.col_min is not None and self.col_max is not None:
            range_val = self.col_max - self.col_min
            range_val[range_val == 0] = 1.0
            input_sample = (input_sample - self.col_min) / range_val
            
        self.forward_feeding(input_sample)
        return self.OL_after_activation.flat[0]

if __name__ == "__main__":
    pass
