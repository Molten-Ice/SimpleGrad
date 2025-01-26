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
    def __init__(self, input_size, output_size, activation = 'sigmoid', no_init = False):
        self.w = Parameter(Tensor.randn(output_size, input_size), _op='w')
        self.b = Parameter(Tensor.randn(output_size, 1), _op='b')
        self.act = activation
        self.input_size = input_size
        self.output_size = output_size


        # Initialize weights (biases already initialized to 0).
        if no_init:
            pass
        elif activation == 'sigmoid':
            std = math.sqrt(1/input_size) # Xavier/Glorot initialization
            self.w.data = std * self.w.data # Don't include in autograd graph
            print(f'Sigmoid initialized for layer {input_size} -> {output_size} (std: {std:.5f})')
        elif activation == 'relu':
            std = math.sqrt(2/input_size)  # He initialization
            self.w.data = std * self.w.data
            print(f'ReLU initialized for layer {input_size} -> {output_size} (std: {std:.5f})')
        else:
            raise ValueError(f"{activation} activation function initialization not supported.")
    
    def forward(self, x):
        z = self.w @ x + self.b
        return getattr(z, self.act)() if self.act is not None else z

    def parameters(self):
        return [self.w, self.b]
    
    def __repr__(self):
        return f"Linear({self.input_size} -> {self.output_size}, activation={self.act})"

class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for n in self.layers for p in n.parameters()]
    
    def __repr__(self):
        content = ',\n'.join(f'   {layer}' for layer in self.layers)
        return f"Sequential(\n{content}\n)"
