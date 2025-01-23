import torch
import numpy as np
import random

class Tensor():
    float32 = torch.float32
    int64 = torch.int64

    def __init__(self, data, dtype=None, device=None):
        # Convert input to PyTorch tensor if it isn't already
        self.device = device
        if isinstance(data, torch.Tensor):
            # print("Warning: Data is already a PyTorch tensor")
            if dtype and data.dtype != dtype:
                print(f"Warning: Data dtype {data.dtype} does not match requested dtype {dtype}, converting to {dtype}")
                self.data = data.to(dtype=dtype)
            else:
                self.data = data
        else:
            if dtype and device:
                self.data = torch.tensor(data, dtype=dtype, device=device)
            elif dtype:
                self.data = torch.tensor(data, dtype=dtype)
            elif device:
                self.data = torch.tensor(data, device=device)
            else:
                self.data = torch.tensor(data)

    def __repr__(self):
        return f"Tensor({self.data})"
    
    # Basic operations
    def __add__(self, other):
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data + other_tensor, device=self.device)
    
    def __mul__(self, other):
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data * other_tensor, device=self.device)
    
    def __matmul__(self, other):
        """Matrix multiplication using @ operator."""
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(torch.matmul(self.data, other_tensor), device=self.device)
    
    # def dot(self, other):
    #     """Alias for matrix multiplication, similar to numpy's dot."""
    #     return self @ other
    
    # Common tensor operations
    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape), device=self.device)
    
    def sum(self, dim=None, keepdim=False):
        return Tensor(self.data.sum(dim=dim, keepdim=keepdim), device=self.device)
    
    def mean(self, dim=None, keepdim=False):
        return Tensor(self.data.mean(dim=dim, keepdim=keepdim), device=self.device)
    
    # Gradient-related methods
    def backward(self):
        self.data.backward()
    
    def grad(self):
        return Tensor(self.data.grad, device=self.device) if self.data.grad is not None else None
    
    # Shape and device properties
    @property
    def shape(self):
        return self.data.shape
    
    # Static methods for tensor creation
    @staticmethod
    def zeros(*shape, dtype=None, device=None):
        return Tensor(torch.zeros(*shape, dtype=dtype, device=device), device=device)
    
    @staticmethod
    def ones(*shape, dtype=None, device=None):
        return Tensor(torch.ones(*shape, dtype=dtype, device=device), device=device)
    
    @staticmethod
    def randn(*shape, device=None):
        return Tensor(torch.randn(*shape, device=device), device=device)
    
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
        return Tensor(self.data - other_tensor, device=self.device)
    
    def __rsub__(self, other):
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(other_tensor - self.data, device=self.device)
    
    def __truediv__(self, other):
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data / other_tensor, device=self.device)
    
    def __rtruediv__(self, other):
        other_tensor = other.data if isinstance(other, Tensor) else other
        return Tensor(other_tensor / self.data, device=self.device)
    
    def exp(self):
        """Returns e raised to the power of each element in the tensor."""
        return Tensor(torch.exp(self.data), device=self.device)
    
    def __neg__(self):
        """Returns the negation of the tensor."""
        return Tensor(-self.data, device=self.device)

    def argmax(self, dim=None):
        """Returns the indices of the maximum values along a dimension.
        If dim is None, the tensor is flattened before finding the argmax."""
        return Tensor(torch.argmax(self.data, dim=dim), device=self.device)

    def transpose(self, dim0=0, dim1=1):
        """Returns a tensor that is a transposed version of the input."""
        return Tensor(self.data.transpose(dim0, dim1), device=self.device)
    
    def T(self):
        """Alias for transpose."""
        return self.transpose()
    
    def to(self, device):
        """Move tensor to specified device (e.g. 'cuda' or 'cpu')"""
        if not device:
            print("No device specified, returning original tensor")
            return self
            
        # Check if tensor is already on the requested device
        if self.device == device:
            return self
            
        self.device = device
        return Tensor(self.data.to(device), device=device)
    
    # Helper methods
    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def tolist(self):
        """Convert tensor to a Python list."""
        return self.data.cpu().detach().numpy().tolist()
    
    def numpy(self):
        """Convert tensor to a numpy array."""
        return self.data.cpu().detach().numpy()
    
    def unsqueeze(self, dim):
        """Add a dimension of size 1 at the specified position."""
        return Tensor(self.data.unsqueeze(dim), device=self.device)

    def squeeze(self, dim=None):
        """Remove dimensions of size 1 from the tensor.
        
        Args:
            dim (int, optional): If given, only squeeze the dimension specified.
                               If None, remove all dimensions of size 1.
        """
        return Tensor(self.data.squeeze(dim), device=self.device)

    @staticmethod
    def stack(tensors, dim=0):
        """Stack a sequence of tensors along a new dimension.
        
        Args:
            tensors: sequence of tensors to stack
            dim: dimension to insert. Default: 0
        """
        # Convert any non-Tensor elements to Tensors first
        tensor_data = [t.data if isinstance(t, Tensor) else torch.tensor(t) for t in tensors]
        # Get the device from the first tensor (if any have one)
        device = next((t.device for t in tensors if isinstance(t, Tensor) and t.device), None)
        return Tensor(torch.stack(tensor_data, dim=dim), device=device)

