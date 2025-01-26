import math
from .tensor import Tensor
from .parameter import Parameter

class Module():
    def __call__(self, x):
        return self.forward(x)
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def parameters(self):
        return []

class Linear(Module):
    """
    w ─┐
       @ ─┐
    x ─┘  │
          + ── z ── φ ── a
    b ────┘
    """
    def __init__(self, input_size, output_size):
        self.w = Parameter(Tensor.randn(output_size, input_size), _op='w')
        self.b = Parameter(Tensor.randn(output_size, 1), _op='b')

        # Initialize weights (biases already initialized to 0).
        std = math.sqrt(1/input_size)  # shape[1] is n_in
        self.w.data = std * self.w.data # Don't include in autograd graph
    
    def forward(self, x):
        z = self.w @ x + self.b
        return z.sigmoid()

    def parameters(self):
        return [self.w, self.b]

class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for n in self.layers for p in n.parameters()]
