from .tensor import Tensor
from .tensor import TorchTensor, NumpyTensor # Only used for type checking.

class Parameter(): # micrograd but for custom Tensor class.
    def __init__(self, data: Tensor, _children=(), _op='', is_weight=False):
        assert isinstance(data, (Tensor, TorchTensor, NumpyTensor)), f"data must be a Tensor (type: {type(data)}), op: {_op}, data: {data}"
        self.data = data
        self.grad = None # Now a matrix not a scalar.
        # internal variables used for autograd graph construction
        self._backward = lambda: None

        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self.is_weight = is_weight

    def _accumulate(self, value):
        """Helper method to initialize grad if None or accumulate if existing"""

        if self.grad is None:
            self.grad = value
        else:
            self.grad += value

    def __add__(self, other):
        other = map_other_to_parameter(other, self.data.device)
        out = Parameter(self.data + other.data, (self, other), '+')

        def _backward():
            self._accumulate(out.grad)
            other._accumulate(out.grad)
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = map_other_to_parameter(other, self.data.device)
        out = Parameter(self.data * other.data, (self, other), '*')

        def _backward():
            self._accumulate(other.data * out.grad)
            other._accumulate(self.data * out.grad)
        out._backward = _backward
        
        return out
    
    def __matmul__(self, other):
        out = Parameter(self.data @ other.data, (self, other), '@')
        def _backward():
            self._accumulate(out.grad @ other.data.transpose(-2, -1))
            other._accumulate(self.data.transpose(-2, -1) @ out.grad)
        out._backward = _backward
        return out
    
    def sum(self):
        out = Parameter(self.data.sum(), (self,), 'sum')
        def _backward():
            # Gradient flows back equally to all input elements
            # Broadcasting the scalar gradient to match input shape
            self._accumulate(Tensor.ones_like(self.data, device=self.data.device) * out.grad)
        out._backward = _backward
        return out
    
    # Could move the logic for L2 regularization into this function.
    def mse(self, target): # Feel like .mean() here isn't correctly backpropagated.
        batch_size = self.data.shape[0]
        out = Parameter(0.5 * (((self.data - target.data)**2).sum())/ batch_size, (self,), 'mse')
        def _backward():
            self._accumulate((self.data - target.data)* out.grad / batch_size)
        out._backward = _backward
        return out
    
    def cross_entropy(self, target, eps=1e-7):
        batch_size = self.data.shape[0]

        # eps is small constant to prevent log(0)
        clipped_data = self.data.clip(eps, 1 - eps)
        out = Parameter(Tensor.nan_to_num(-target.data * clipped_data.log() - (1-target.data) * (1-clipped_data).log()).sum() / batch_size, (self,), 'cross_entropy')

        def _backward():
            grad = (clipped_data - target.data) / (clipped_data * (1 - clipped_data))
            self._accumulate(grad * out.grad / batch_size)
        out._backward = _backward
        return out


    def sigmoid(self):
        out = Parameter(1.0/(1.0 + (-self.data).exp()), (self,), 'sigmoid')
        def _backward():
            self._accumulate((1-out.data) * out.data * out.grad)
        out._backward = _backward
        return out
    
    def relu(self):
        out = Parameter(self.data * (self.data > 0.0), (self,), 'relu')
        def _backward():
            self._accumulate((out.data > 0.0) * out.grad)
        out._backward = _backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    # reflected/reversed arithmetic operations,
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Parameter(data={self.data}, grad={self.grad})"
    
    def to(self, device):
        self.data = self.data.to(device)
    
    @property
    def shape(self):
        return self.data.shape if self.data is not None else None
    

def map_other_to_parameter(other, device=None):
    if isinstance(other, Parameter):
        return other
    elif isinstance(other, (int, float)):
        return Parameter(Tensor(other, device=device))
    elif isinstance(other, (Tensor, TorchTensor, NumpyTensor)):
        return Parameter(Tensor(other, device=device))
    raise ValueError(f"map_other_to_parameter | invalid type: {type(other)}")
