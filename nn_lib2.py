import torch
from typing import List

from kernels.feedforward import feedforward_op
from kernels.add import add
from kernels.subtract import subtract
from kernels.point_multiply import point_multiply
from kernels.matmul import matmul

from functions.loss import least_squares, least_squares_deriv
from functions.activation import sigmoid, sigmoid_deriv, relu, relu_deriv, leaky_relu, leaky_relu_deriv, tanh, tanh_deriv

class NeuralNetwork:
    def __init__(self, layers: List[(int, str)], loss_f="least_squares", learning_rate=0.01):
        if len(layers) < 2:
            raise Exception("There must be at least 2 layers: input and output")
        
        min_init_weight, max_init_weight = -0.1, 0.1
        self.input_dim = layers[0]
        
        prev_dim = self.input_dim
        self.layers = []
        for layer_dim, actf in layers[1:]:
            weights = torch.rand(layer_dim, prev_dim) * (max_init_weight - min_init_weight) + min_init_weight
            biases = torch.zeros(layer_dim)
            layer = {
                "weights": weights,
                "bias": biases,
                "inputs": None,
                "preactivations": None,
                "dim": layer_dim,
                "actf": actf if actf in ["tanh", "leaky_relu", "relu", "sigmoid"] else None
            }

            self.layers.append(layer)
            prev_dim = layer_dim

        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()
        self.lossd_f = least_squares_deriv
    
    def feed_forward(self, inputs: List[List[float]]):
        if len(inputs) < 1:
            return

        input_matrix = torch.zeros((len(inputs), self.input_dim))
        for i, inp in enumerate(inputs):
            if len(inp) != self.input_dim:
                raise Exception("Dimensions of input[", i, "] must match that of input layer dimension")
        
            input_matrix[i, :] = torch.tensor(inp)
        
        activations = input_matrix
        for layer in self.layers:
            layer["inputs"] = activations.clone().detach()
            weights_transposed = torch.transpose(layer["weights"], 0, 1)
            preactivations, activations = feedforward_op(activations, weights_transposed, layer["bias"])
            layer["preactivations"] = preactivations.clone().detach()
        
        return activations
    
    def backpropagate(self, losses):
        if losses.shape[-1] != self.layers[-1]["dim"]:
            raise Exception("losses must match output layer dimensions")

        delta = self.lossd_f(losses)

        for layer in reversed(self.layers):
            old_weights = layer["weights"].clone()
            act_deriv = self.get_activationfn_deriv(layer.get("actf"), layer["preactivations"])
            delta = point_multiply(delta, act_deriv)
            weight_grad = matmul(delta.transpose(0, 1), layer["inputs"])
            bias_grad = delta.sum(dim=0) if delta.ndim > 1 else delta
            
            layer["weights"] = subtract(layer["weights"], weight_grad, self.learning_rate)
            layer["bias"] = subtract(layer["bias"], bias_grad, self.learning_rate)

            delta = matmul(delta, old_weights)
        
    
    def get_activationfn():
        if layer["actf"] == "relu":
            return relu(preactivations)
        elif layer["actf"] == "leaky_relu":
            return leaky_relu(preactivations, alpha=0.01)
        elif layer["actf"] == "sigmoid":
            return sigmoid(preactivations)
        elif layer["actf"] == "tanh":
            return tanh(preactivations)
        else:
            return preactivations

    def get_activationfn_deriv(self, actf, preactivations: torch.Tensor) -> torch.Tensor:
        if actf is None:
            return torch.ones_like(preactivations)
        elif actf == "leaky_relu":
            return leaky_relu_deriv(preactivations, alpha=0.01)
        elif actf == "relu":
            return relu_deriv(preactivations)
        elif actf == "sigmoid":
            return sigmoid_deriv(preactivations)
        elif actf == "tanh":
            return torch_tanh_deriv(preactivations)
        else:
            return torch.ones_like(preactivations)


if __name__ == "__main__":
    layers = [(1, None), (2, "relu"), (3, "wet")]
    nn = NeuralNetwork(layers)
    ins = [[5]]
    out = nn.feed_forward(ins)
    print(out)
        
