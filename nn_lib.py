import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from functions.activation_functions import tanh, tanh_deriv, leaky_relu, leaky_relu_deriv, relu, relu_deriv, sigmoid, sigmoid_deriv
from functions.loss_functions import least_squares, least_squares_deriv


#Neuron Types
class InputNeuron:
    def __init__(self):
        self.input = 0
    
    def calc_grad(gradient):
        raise NotImplementedError

class HiddenNeuron:
    def __init__(self, weights, bias, actf, actf_d):
        self.weights = weights
        self.bias = bias
        self.actf = actf
        self.actf_d = actf_d
        self.inputs = []

    def calc_output(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("inputs must be same size as weights")
    
        self.inputs = inputs

        out = 0
        for i in range(len(inputs)):
            out += inputs[i] * self.weights[i]
        
        out += self.bias
        return self.actf(out)

    
    def calc_grad(gradient):
        raise NotImplementedError

class OutputNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.inputs = []
    
    def calc_grad(gradient):
        raise NotImplementedError
    
    def calc_output(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("inputs must be same size as weights")
    
        self.inputs = inputs

        out = 0
        for i in range(len(inputs)):
            out += inputs[i] * self.weights[i]
        
        out += self.bias
        return out #no activation function in output layer

#Neural Networks
class NeuralNetwork:
    def __init__(self, in_width, out_width, layers_init, learning_rate=0.01):
        input_layer = [InputNeuron() for _ in range(in_width)]
        hidden_layers = []
        prev_layer_width = in_width
        for layer in layers_init:
            new_layer = []
            layer_width, actf_str = layer
            actf, actf_d = self.get_act_func(actf_str)

            for _ in range(layer_width):
                init_weights = [random.uniform(-0.1, 0.1) for _ in range(prev_layer_width)]
                init_bias = 0
                new_layer.append(HiddenNeuron(init_weights, init_bias, actf, actf_d))
            
            hidden_layers.append(new_layer)
            prev_layer_width = layer_width

        init_out_weights = [random.uniform(-0.1, 0.1) for _ in range(prev_layer_width)]
        init_out_bias = 0
        output_layer = [OutputNeuron(init_out_weights, init_out_bias) for i in range(out_width)]

        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

        self.learning_rate = learning_rate
        self.lossf = least_squares
        self.lossf_d = least_squares_deriv
    
    def get_act_func(self, actf_str):
        if actf_str == "tanh":
            return tanh, tanh_deriv

        if actf_str == "leaky_relu":
            return leaky_relu, leaky_relu_deriv
        
        if actf_str == "relu":
            return relu, relu_deriv
        
        if actf_str == "sigmoid":
            return sigmoid, sigmoid_deriv

        raise ValueError(f'activation string {actf_str }does not correspond to any activation function')

    
    def feed_forward(self, inputs):
        if len(inputs) != len(self.input_layer):
            raise ValueError("inputs must be same size as input_layer")
        
        for i in range(len(inputs)):
            self.input_layer[i].input = inputs[i]

        cur_layer_inputs = []
        for neuron in self.input_layer:
            cur_layer_inputs.append(neuron.input)
        
        for h_layer in self.hidden_layers:
            next_layer_inputs = []
            for neuron in h_layer:
                next_layer_inputs.append(neuron.calc_output(cur_layer_inputs))
            
            cur_layer_inputs = next_layer_inputs
            next_layer_inputs = []
        
        outputs = []
        for out_neuron in self.output_layer:
            outputs.append(out_neuron.calc_output(cur_layer_inputs))
        
        return outputs

    #Implemented for single example Stochastic Gradient Descent
    def backprop(self, inputs, y_true):
        '''
        weight_backprop(cur_deriv):
            grad = 0
            weight_num;
            intermed_outupt = w1x+w2x+...+b
            for neuron in previous_layer:
                grad += cur_deriv * activ_deriv(intermed_outupt) * neuron.weights[weight_num] * self.input
            
            return grad

        '''
        if len(inputs) != len(self.input_layer) or len(y_true) != len(self.output_layer):
            raise ValueError("inputs must be same size as input layer and truth outputs must be same size as output layer")

        y_pred = self.feed_forward(inputs)
        loss_d = []
        for i in range(len(y_pred)):
            loss_d.append(self.lossf_d([y_pred[i]], [y_true[i]]))

        if self.hidden_layers:
            prev_layer_derivs = [0 for _ in range(len(self.hidden_layers[-1]))]

        #Update output layer
        for neuron_i, neuron in enumerate(self.output_layer):
            #update bias
            bias_grad = loss_d[neuron_i]
            neuron.bias -= bias_grad * self.learning_rate

            #update weights
            for weight_i, weight in enumerate(neuron.weights):
                if prev_layer_derivs:
                    prev_layer_derivs[weight_i] += loss_d[neuron_i] * weight #sum up derivates for each respective input neuron from the previous layer across all neurons
                weight_grad = loss_d[neuron_i] * neuron.inputs[weight_i]
                neuron.weights[weight_i] -= weight_grad * self.learning_rate
            

        #Update hidden layers
        for hlayer_i in range(len(self.hidden_layers) - 1, -1, -1):
            current_layer_derivs = prev_layer_derivs

            if hlayer_i > 0:
                prev_layer_derivs = [0 for _ in range(len(self.hidden_layers[hlayer_i-1]))]
            
            for neuron_i, neuron in enumerate(self.hidden_layers[hlayer_i]):
                #Calculate w1*x1 + w2*x2 + ... + b
                weighted_input = neuron.bias
                for weight_i in range(len(neuron.weights)):
                    weighted_input += neuron.weights[weight_i] * neuron.inputs[weight_i]
                
                actf_d_calc = neuron.actf_d(weighted_input)

                #update bias
                bias_grad = current_layer_derivs[neuron_i] * actf_d_calc
                neuron.bias -= bias_grad * self.learning_rate
            
                #update weights
                for weight_i, weight in enumerate(neuron.weights):
                    if hlayer_i > 0:
                        prev_layer_derivs[weight_i] += current_layer_derivs[neuron_i] * actf_d_calc * weight #sum up derivates for each respective input neuron from the previous layer across all neurons
                    
                    weight_grad = current_layer_derivs[neuron_i] * actf_d_calc * neuron.inputs[weight_i]
                    neuron.weights[weight_i] -= weight_grad * self.learning_rate



    #Calculate
    def train(self, X, Y, epochs):
        progress_bar = tqdm(range(epochs), desc='Training', unit='epoch')
        for _ in progress_bar:
            for i in range(len(X)):
                self.backprop(X[i], Y[i])

def draw_model_2d(nn: NeuralNetwork, start, end, step=0.1, points=[]):
    if len(nn.output_layer) != 1 or len(nn.input_layer) != 1:
        raise ValueError("Input and output layers must have dim 1 to draw 2d")
        
    num_steps = int((end - start) / step)
    inferences = []

    for i in range(num_steps):
        x = (i * step) + start
        inferences.append([x, nn.feed_forward([x])[0]])


    x, y = zip(*inferences)

    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    x_points, y_points = zip(*points)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, '-', label='OG', color='green')  # Smooth curve
    plt.plot(x_points, y_points, 'o', label='Additional Points', color='red')  # Additional points
    #plt.plot(x_smooth, y_smooth, label='Smooth Curve', color='blue')  # Smooth curve
    plt.title("Model Trained on Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.show()