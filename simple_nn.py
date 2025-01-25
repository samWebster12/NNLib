import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

#Using a simple one neuron input + 3 neuron hidden layer + 1 neuron output
#Middle neuron was added later hence the out of order namings 

'''
           w1          w3
          /--->  b1   -----  
         /                 \
        /  w5          w6   \
Input  ------>   b4   ------->  b3 --> Output
        \                   /
         \  w2         w4  /
          \--->  b2   -----

'''

#Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    """Calculate ReLU of x."""
    return max(0, x)

def relu_deriv(x):
    """Calculate the derivative of ReLU at x."""
    return 1 if x > 0 else 0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha*x

def leaky_relu_deriv(x, alpha=0.01):
    return 1 if x > 0 else alpha


#Simple Neural Network
class NeuralNetwork:
    def __init__(self, activation, activation_deriv):
        self.w1 = random.uniform(-0.1, 0.1)
        self.w2 = random.uniform(-0.1, 0.1)
        self.w3 = random.uniform(-0.1, 0.1)
        self.w4 = random.uniform(-0.1, 0.1)
        self.w5 = random.uniform(-0.1, 0.1)
        self.w6 = random.uniform(-0.1, 0.1)
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0
        self.activation = activation
        self.activation_deriv = activation_deriv


    def calc_b3_gradient(self, actual, predicted):
        if len(actual) != len(predicted):
            raise IndexError("actual size different from predicted")
        
        grad = 0
        for i in range(len(actual)):
            y = actual[i][1]
            predicted_y = predicted[i][1]

            grad += -2 * (y - predicted_y)
        
        return grad

    def calc_w3_gradient(self, actual, predicted):
        if len(actual) != len(predicted):
            raise IndexError("actual size different from predicted")
        
        grad = 0
        for i in range(len(actual)):
            x = actual[i][0]
            y = actual[i][1]
            predicted_y = predicted[i][1]

            grad += -2 * (y - predicted_y) * self.activation(self.w1*x + self.b1)
        
        return grad

    def calc_w4_gradient(self, actual, predicted):
        if len(actual) != len(predicted):
            raise IndexError("actual size different from predicted")
        
        grad = 0
        for i in range(len(actual)):
            x = actual[i][0]
            y = actual[i][1]
            predicted_y = predicted[i][1]

            grad += -2 * (y - predicted_y) * self.activation(self.w2*x + self.b2)
        
        return grad

    def calc_w6_gradient(self, actual, predicted):
        if len(actual) != len(predicted):
            raise IndexError("actual size different from predicted")
        
        grad = 0
        for i in range(len(actual)):
            x = actual[i][0]
            y = actual[i][1]
            predicted_y = predicted[i][1]

            grad += -2 * (y - predicted_y) * self.activation(self.w5*x + self.b4)
        
        return grad

    def calc_b1_gradient(self, actual, predicted):
        if len(actual) != len(predicted):
            raise IndexError("actual size different from predicted")
        
        grad = 0
        for i in range(len(actual)):
            x = actual[i][0]
            y = actual[i][1]
            predicted_y = predicted[i][1]

            grad += -2 * (y - predicted_y) * (self.w3 * self.activation_deriv(self.w1*x + self.b1))
        
        return grad


    def calc_b2_gradient(self, actual, predicted):
        if len(actual) != len(predicted):
            raise IndexError("actual size different from predicted")
        
        grad = 0
        for i in range(len(actual)):
            x = actual[i][0]
            y = actual[i][1]
            predicted_y = predicted[i][1]

            grad += -2 * (y - predicted_y) * (self.w4 * self.activation_deriv(self.w2*x + self.b2))
        
        return grad
    
    def calc_b4_gradient(self, actual, predicted):
        if len(actual) != len(predicted):
            raise IndexError("actual size different from predicted")
        
        grad = 0
        for i in range(len(actual)):
            x = actual[i][0]
            y = actual[i][1]
            predicted_y = predicted[i][1]

            grad += -2 * (y - predicted_y) * (self.w6 * self.activation_deriv(self.w5*x + self.b4))
        
        return grad

    def calc_w1_gradient(self, actual, predicted):
        if len(actual) != len(predicted):
            raise IndexError("actual size different from predicted")
        
        grad = 0
        for i in range(len(actual)):
            x = actual[i][0]
            y = actual[i][1]
            predicted_y = predicted[i][1]

            grad += -2 * (y - predicted_y) * (self.w3 * self.activation_deriv(self.w1*x + self.b1) * x)
        
        return grad

    def calc_w2_gradient(self, actual, predicted):
        if len(actual) != len(predicted):
            raise IndexError("actual size different from predicted")
        
        grad = 0
        for i in range(len(actual)):
            x = actual[i][0]
            y = actual[i][1]
            predicted_y = predicted[i][1]

            grad += -2 * (y - predicted_y) * (self.w4 * self.activation_deriv(self.w2*x + self.b2) * x)
        
        return grad

    def calc_w5_gradient(self, actual, predicted):
        if len(actual) != len(predicted):
            raise IndexError("actual size different from predicted")
        
        grad = 0
        for i in range(len(actual)):
            x = actual[i][0]
            y = actual[i][1]
            predicted_y = predicted[i][1]

            grad += -2 * (y - predicted_y) * (self.w6 * self.activation_deriv(self.w5*x + self.b4) * x)
        
        return grad

    def train(self, training_data, iterations, learning_rate):
        for i in range(iterations):
            #Calculate predicted
            predicted = []
            for i in range(len(training_data)):
                predicted.append([0, self.infer(training_data[i][0])])

            w1_grad = self.calc_w1_gradient(training_data, predicted)
            w2_grad = self.calc_w2_gradient(training_data, predicted)
            w3_grad = self.calc_w3_gradient(training_data, predicted)
            w4_grad = self.calc_w4_gradient(training_data, predicted)
            w5_grad = self.calc_w5_gradient(training_data, predicted)
            w6_grad = self.calc_w6_gradient(training_data, predicted)
            b1_grad = self.calc_b1_gradient(training_data, predicted)
            b2_grad = self.calc_b2_gradient(training_data, predicted)
            b3_grad = self.calc_b3_gradient(training_data, predicted)
            b4_grad = self.calc_b4_gradient(training_data, predicted)


            self.w1 = self.w1 - w1_grad * learning_rate
            self.w2 = self.w2 - w2_grad * learning_rate
            self.w3 = self.w3 - w3_grad * learning_rate
            self.w4 = self.w4 - w4_grad * learning_rate
            self.w5 = self.w5 - w5_grad * learning_rate
            self.w6 = self.w6 - w6_grad * learning_rate
            self.b1 = self.b1 - b1_grad * learning_rate
            self.b2 = self.b2 - b2_grad * learning_rate
            self.b3 = self.b3 - b3_grad * learning_rate
            self.b4 = self.b4 - b4_grad * learning_rate

    def infer(self, x):
        return self.b3 + (self.w3 * (self.activation(self.w1*x + self.b1))) + (self.w4 * (self.activation(self.w2*x + self.b2))) + (self.w6 * (self.activation(self.w5*x + self.b4)))

    def draw_model(self, start, end, step=0.1, additional_points=[]):
        num_steps = int((end - start) / step)
        inferences = []

        for i in range(num_steps):
            x = (i * step) + start
            inferences.append([x, nn.infer(x)])


        x, y = zip(*inferences)

        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)

        x_additional, y_additional = zip(*additional_points)

        # Create a smooth curve
        x_smooth = np.linspace(x.min(), x.max(), 300)  # Fine x-axis for a smooth curve
        spl = make_interp_spline(x, y, k=3)  # Cubic spline
        y_smooth = spl(x_smooth)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, '-', label='OG', color='green')  # Smooth curve
        plt.plot(x_additional, y_additional, 'o', label='Additional Points', color='red')  # Additional points
        #plt.plot(x_smooth, y_smooth, label='Smooth Curve', color='blue')  # Smooth curve
        plt.title("Smooth Curve Through Points")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.grid(True)
        plt.show()


#Generate random points around the line 2x + 3
x_values = np.linspace(-10, 10, 20)  # 20 points between -10 and 10
y_values = 2 * x_values + 3 + np.random.normal(0, 2, size=x_values.shape)  # Line with noise
training_data = np.column_stack((x_values, y_values))

learning_rate = 0.0001
its = 10000

nn = NeuralNetwork(relu, relu_deriv)
nn.train(training_data, its, learning_rate)
nn.draw_model(-10, 10, additional_points=training_data)