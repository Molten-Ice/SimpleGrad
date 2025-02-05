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
    
    def train(self):
        self.training = True
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.train()
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        item.train()
    
    def eval(self):
        self.training = False
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.eval()
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        item.eval()
    

class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
    
        mask = (Tensor.randn(*x.shape) > self.p).astype(Tensor.float32)
        scale = 1.0 / (1.0 - self.p)

        mask = Parameter(mask, _op='mask')
        return x * mask * scale


    def __repr__(self):
        return f"Dropout(p={self.p}, training={self.training})"

class Linear(Module):
    """
    w ─┐
       @ ─┐
    x ─┘  │
          + ── z ── φ ── a
    b ────┘
    Input -> Linear (Wx + b) -> Activation -> Dropout -> Next Layer
    """
    def __init__(self, input_size, output_size, activation = 'sigmoid', dropout = 0, no_init = False):
        self.w = Parameter(Tensor.randn(output_size, input_size), _op='w', is_weight=True)
        self.b = Parameter(Tensor.randn(output_size, 1), _op='b')
        self.act = activation
        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.input_size = input_size
        self.output_size = output_size

        print(f'Linear layer {input_size} -> {output_size}, activation: {activation}, dropout: {dropout}' + ' no init' if no_init else '')
        # Initialize weights (biases already initialized to 0).
        if no_init:
            pass
        elif activation == None:
            print(f'No activation for layer {input_size} -> {output_size}')
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
        a = getattr(z, self.act)() if self.act is not None else z
        return self.dropout(a) if self.dropout else a

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
