from .tensor import Tensor

class Parameter(): # micrograd but for custom Tensor class.
    def __init__(self, data: Tensor, _children=(), _op=''):
        self.data = data
        self.grad = None # Now a matrix not a scalar.
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def _accumulate(self, value):
        """Helper method to initialize grad if None or accumulate if existing"""
        if self.grad is None:
            self.grad = value
        else:
            self.grad += value

    def __add__(self, other):
        other = other if isinstance(other, Parameter) else Parameter(other)
        out = Parameter(self.data + other.data, (self, other), '+')

        def _backward():
            self._accumulate(out.grad)
            other._accumulate(out.grad)
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        out = Parameter(self.data @ other.data, (self, other), '@')
        def _backward():
            self._accumulate(out.grad @ other.data.transpose(-2, -1))
            other._accumulate(self.data.transpose(-2, -1) @ out.grad)
        out._backward = _backward
        return out
    

    def mse(self, target):
        out = Parameter(0.5 * ((self.data - target.data) ** 2).mean(), (self,), 'mse')
        def _backward():
            self._accumulate((self.data - target.data) * out.grad)
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
        print(f'{other} * {self}')
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Parameter(data={self.data}, grad={self.grad})"
    
