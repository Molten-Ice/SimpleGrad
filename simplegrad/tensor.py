import torch
import numpy as np
import random

class Tensor():
    def __init__(self, data):
        # Convert input to PyTorch tensor if it isn't already
        self.data = torch.tensor(data) if not isinstance(data, torch.Tensor) else data
    
    def __repr__(self):
        return f"Tensor({self.data})"
    
    # Basic operations
    def __add__(self, other):
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data + other_tensor)
    
    def __mul__(self, other):
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data * other_tensor)
    
    def __matmul__(self, other):
        """Matrix multiplication using @ operator."""
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(torch.matmul(self.data, other_tensor))
    
    # def dot(self, other):
    #     """Alias for matrix multiplication, similar to numpy's dot."""
    #     return self @ other
    
    # Common tensor operations
    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape))
    
    def sum(self, dim=None, keepdim=False):
        return Tensor(self.data.sum(dim=dim, keepdim=keepdim))
    
    def mean(self, dim=None, keepdim=False):
        return Tensor(self.data.mean(dim=dim, keepdim=keepdim))
    
    # Gradient-related methods
    def backward(self):
        self.data.backward()
    
    def grad(self):
        return Tensor(self.data.grad) if self.data.grad is not None else None
    
    # Shape and device properties
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def device(self):
        return self.data.device
    
    # Static methods for tensor creation
    @staticmethod
    def zeros(*shape):
        return Tensor(torch.zeros(*shape))
    
    @staticmethod
    def ones(*shape):
        return Tensor(torch.ones(*shape))
    
    @staticmethod
    def randn(*shape):
        return Tensor(torch.randn(*shape))
    
    @staticmethod
    def set_seed(seed=3):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    # Right-side operations
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data - other_tensor)
    
    def __rsub__(self, other):
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(other_tensor - self.data)
    
    def __truediv__(self, other):
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data / other_tensor)
    
    def __rtruediv__(self, other):
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(other_tensor / self.data)
    
    def exp(self):
        """Returns e raised to the power of each element in the tensor."""
        return Tensor(torch.exp(self.data))
    
    def __neg__(self):
        """Returns the negation of the tensor."""
        return Tensor(-self.data)

    def argmax(self, dim=None):
        """Returns the indices of the maximum values along a dimension."""
        return int(torch.argmax(self.data, dim=dim).item()) if dim is not None else int(torch.argmax(self.data).item())

    def transpose(self, dim0=0, dim1=1):
        """Returns a tensor that is a transposed version of the input."""
        return Tensor(self.data.transpose(dim0, dim1))
    
    def T(self):
        """Alias for transpose."""
        return self.transpose()
