# NNLib — Building Neural Networks from Scratch

> **Hands‑on playground for understanding neural‑net internals.** From a single‑neuron demo to a Triton‑accelerated matrix‑mult, every directory captures a new learning milestone.

---

## 📁 Project layout

```
NNLib/
├── nn.py                # ⟵ runnable demo using v1 API
├── simple_nn.py         # ⟵ very first single‑hidden‑layer experiment
├── nn_lib.py            # ⟵ v1: pure‑Python / NumPy implementation
├── nn_lib2.py           # ⟵ v2: Torch + Triton kernels
├── functions/           # numerical utils
│   ├── activation.py    #   activation fn & derivatives (Torch)
│   └── loss.py          #   least‑squares + derivative
├── kernels/             # Triton GPU kernels used by v2
│   ├── activations.py   #   element‑wise activations
│   ├── add.py           #   vector addition
│   ├── feedforward.py   #   tiled MatMul + bias + activation
│   ├── matmul.py        #   baseline matmul (WIP)
│   ├── point_multiply.py
│   └── subtract.py
└── TODO.txt
```

## 🔧 Installation

### 1. Clone & create env

```bash
$ git clone https://github.com/<you>/NNLib.git
$ cd NNLib
$ python -m venv .venv && source .venv/bin/activate
```

### 2. Pick your flavour

| Want to run…                       | Install                                                                |
| ---------------------------------- | ---------------------------------------------------------------------- |
| only `simple_nn.py` or `nn_lib.py` | `pip install numpy matplotlib tqdm`                                    |
| `nn_lib2.py` on **CPU**            | `pip install torch numpy matplotlib tqdm`                              |
| `nn_lib2.py` on **NVIDIA GPU**     | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
|                                    | `pip install triton`                                                   |

> **note:** Skip Triton if you’re not on a CUDA‑capable GPU

---

## 🚀 Quick start

### Pure NumPy network (v1)

Run it:

```bash
$ python nn.py
```

A window pops up showing how the learned curve fits the data.

### Triton‑accelerated network (v2)

```python
from nn_lib2 import NeuralNetwork

net = NeuralNetwork(layers=[(2, None), (64, "leaky_relu"), (32, "leaky_relu"), (1, None)],
                    learning_rate=1e‑3)

# batch of 128 random samples
import torch, math
X = torch.rand(128, 2)
y_true = torch.sin(X[:,0]*math.pi) + torch.cos(X[:,1]*math.pi)

for _ in range(2000):
    y_pred = net.feed_forward(X.tolist())
    loss   = (y_pred.squeeze() - y_true).pow(2).mean()
    net.backpropagate(loss.unsqueeze(0))
```
---

## 🧠 What I learned building this

- **Manual back‑prop math**: starting with per‑parameter gradients before vectorising.
- **Stability tricks**: small‑weight initialisation & learning‑rate scheduling.
- **GPU kernels**: how tiled MatMul, bias broadcast, and activation fuse for memory efficiency.

---
