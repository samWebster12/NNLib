import torch

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

def sigmoid_deriv(x: torch.Tensor) -> torch.Tensor:
    s = sigmoid(x)
    return s * (1 - s)

def relu(x: torch.Tensor) -> torch.Tensor:
    # You could also use torch.relu(x)
    return torch.maximum(torch.zeros_like(x), x)

def relu_deriv(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).float()

def leaky_relu(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    return torch.where(x > 0, x, alpha * x)

def leaky_relu_deriv(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    return torch.where(x > 0, torch.ones_like(x), torch.full_like(x, alpha))

def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)

def tanh_deriv(x: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.tanh(x) ** 2
