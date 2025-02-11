from nn_lib import NeuralNetwork, draw_model_2d
import numpy as np

input_dim = 1
output_dim = 1
hidden_layers = [(6, "leaky_relu"), (6, "leaky_relu")]
learning_rate = 0.001
epochs = 8000

nn = NeuralNetwork(input_dim, output_dim, hidden_layers, learning_rate)

#Define x space
start_x = -10
stop_x = 10
x_values = np.linspace(start_x, stop_x, stop_x-start_x)  # 20 points between -10 and 10

'''
# Parabolic function: ax² + b with noise
# Using a=0.25 to control the steepness of the parabola
a = 0.25
b = 5
y_values = a * x_values**2 + b + np.random.normal(0, 2, size=x_values.shape)  # Parabola with noise
'''

'''
#Linear function: ax + b with noise
a = 2
b = 3
y_values = a * x_values + b + np.random.normal(0, 2, size=x_values.shape)  # Line with noise
'''
# Sigmoid-like shape
#y_values = 20 / (1 + np.exp(-0.5 * x_values)) + np.random.normal(0, 1, size=x_values.shape)

# Sinusoidal wave
y_values = 10 * np.sin(0.5 * x_values) + np.random.normal(0, 1, size=x_values.shape)

# Combined sinusoidal waves
#y_values = 5 * np.sin(0.5 * x_values) + 3 * np.cos(x_values) + np.random.normal(0, 1, size=x_values.shape)

# Exponential growth
#y_values = 2 * np.exp(0.2 * x_values) + np.random.normal(0, 1, size=x_values.shape)

# Polynomial (cubic function)
#y_values = 0.01 * x_values**3 - 0.3 * x_values**2 + x_values + np.random.normal(0, 1, size=x_values.shape)

# Step-like function using tanh
#y_values = 10 * np.tanh(0.5 * x_values) + np.random.normal(0, 1, size=x_values.shape)

# Damped oscillation
#y_values = 10 * np.exp(-0.1 * np.abs(x_values)) * np.sin(x_values) + np.random.normal(0, 1, size=x_values.shape)

# Square root with oscillation
#y_values = np.sqrt(np.abs(x_values)) * np.sin(x_values) + np.random.normal(0, 1, size=x_values.shape)

x_values = [[val] for val in x_values]
y_values = [[val] for val in y_values]
training_data = np.column_stack((x_values, y_values))

nn.train(x_values, y_values, epochs)
draw_model_2d(nn, start_x, stop_x, points=training_data)


