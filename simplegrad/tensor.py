import random

import torch
import numpy as np

USE_TORCH = False

class NumpyTensor():
    float32 = np.float32
    int64 = np.int64

    def __init__(self, data, dtype=None, device=None):
        if isinstance(data, np.ndarray):
            if dtype and data.dtype != dtype:
                print(f"Warning: Data dtype {data.dtype} does not match requested dtype {dtype}, converting to {dtype}")
                self.data = data.astype(dtype)
            else:
                self.data = data
        else:
            self.data = np.array(data, dtype=dtype)

    def __repr__(self):
        return f"NumpyTensor({self.data})"
    
    # Basic operations
    def __add__(self, other):
        other_NumpyTensor = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data + other_NumpyTensor)
    
    def __mul__(self, other):
        other_NumpyTensor = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data * other_NumpyTensor)
    
    def __matmul__(self, other):
        other_NumpyTensor = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data @ other_NumpyTensor)
    
    # Common Tensor operations
    def reshape(self, *shape):
        return NumpyTensor(self.data.reshape(*shape))
    
    def sum(self, dim=None, keepdims=False):
        return NumpyTensor(self.data.sum(axis=dim, keepdims=keepdims))
    
    def mean(self, dim=None, keepdims=False):
        return NumpyTensor(self.data.mean(axis=dim, keepdims=keepdims))
    
    # Shape property
    @property
    def shape(self):
        return self.data.shape
    
    # Static methods for Tensor creation
    @staticmethod
    def zeros(*shape, dtype=None, device=None):
        # Handle both tuple and unpacked arguments
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return NumpyTensor(np.zeros(shape, dtype=dtype))
    
    @staticmethod
    def ones(*shape, dtype=None, device=None):
        return NumpyTensor(np.ones(shape, dtype=dtype))
    
    @staticmethod
    def randn(*shape, device=None):
        return NumpyTensor(np.random.randn(*shape))
    
    @staticmethod
    def set_seed(seed=3):
        random.seed(seed)
        np.random.seed(seed)
    
    # Right-side operations
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        other_NumpyTensor = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data - other_NumpyTensor)
    
    def __rsub__(self, other):
        other_NumpyTensor = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(other_NumpyTensor - self.data)
    
    def __truediv__(self, other):
        other_NumpyTensor = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data / other_NumpyTensor)
    
    def __rtruediv__(self, other):
        other_NumpyTensor = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(other_NumpyTensor / self.data)
    
    def exp(self):
        return NumpyTensor(np.exp(self.data))
    
    def __neg__(self):
        return NumpyTensor(-self.data)

    def argmax(self, dim=None):
        return NumpyTensor(np.argmax(self.data, axis=dim))

    def transpose(self, dim0=0, dim1=1):
        # Convert negative indices to positive indices
        ndim = len(self.data.shape)
        if dim0 < 0:
            dim0 = ndim + dim0
        if dim1 < 0:
            dim1 = ndim + dim1
        # Create a list of axes where we swap dim0 and dim1
        axes = list(range(ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return NumpyTensor(self.data.transpose(axes))
    
    def T(self):
        return self.transpose()
    
    def tolist(self):
        return self.data.tolist()
    
    def numpy(self):
        return self.data
    
    def unsqueeze(self, dim):
        return NumpyTensor(np.expand_dims(self.data, axis=dim))
    
    def squeeze(self, dim=None):
        return NumpyTensor(np.squeeze(self.data, axis=dim))
    
    @staticmethod
    def stack(tensors, dim=0):
        tensor_data = [t.data if isinstance(t, NumpyTensor) else np.array(t) for t in tensors]
        return NumpyTensor(np.stack(tensor_data, axis=dim))
    
    @staticmethod
    def get_device():
        return None

    def to(self, device):
        """Device parameter added for compatibility, but does nothing for numpy"""
        return self

    def __pow__(self, other):
        other_NumpyTensor = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data ** other_NumpyTensor)
    
    def __rpow__(self, other):
        other_NumpyTensor = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(other_NumpyTensor ** self.data)

    def __lt__(self, other):
        other_data = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data < other_data)

    def __gt__(self, other):
        other_data = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data > other_data)
    
    def astype(self, dtype):
        """Convert tensor to a different dtype"""
        return NumpyTensor(self.data.astype(dtype))

    @staticmethod
    def ones_like(tensor, dtype=None, device=None):
        """Returns a tensor of ones with the same shape as input tensor"""
        if isinstance(tensor, NumpyTensor):
            tensor = tensor.data
        return NumpyTensor(np.ones_like(tensor, dtype=dtype))

    def norm(self):
        """Returns the L2 (Euclidean) norm of the tensor."""
        return NumpyTensor(np.sqrt(np.sum(self.data * self.data)))

    def log(self):
        """Returns the natural logarithm of the tensor elements"""
        return NumpyTensor(np.log(self.data))
    
    @staticmethod
    def nan_to_num(tensor, nan=0.0, posinf=None, neginf=None):
        """Replace NaN, positive infinity, and negative infinity with specified values"""
        if isinstance(tensor, NumpyTensor):
            tensor = tensor.data
        return NumpyTensor(np.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf))

    def clip(self, min_val, max_val):
        """Clips tensor values between min_val and max_val"""
        return NumpyTensor(np.clip(self.data, min_val, max_val))

    def abs(self):
        """Returns the absolute value of each element in the tensor"""
        return NumpyTensor(np.abs(self.data))

class TorchTensor():
    float32 = torch.float32
    int64 = torch.int64

    def __init__(self, data, dtype=None, device=None):
        # Convert input to PyTorch TorchTensor if it isn't already
        self.device = device
        if isinstance(data, torch.Tensor):
            # print("Warning: Data is already a PyTorch TorchTensor")
            if dtype and data.dtype != dtype:
                print(f"Warning: Data dtype {data.dtype} does not match requested dtype {dtype}, converting to {dtype}")
                self.data = data.to(dtype=dtype)
            else:
                self.data = data
        else:
            if dtype and device:
                self.data = torch.Tensor(data, dtype=dtype, device=device)
            elif dtype:
                self.data = torch.Tensor(data, dtype=dtype)
            elif device:
                self.data = torch.Tensor(data, device=device)
            else:
                self.data = torch.Tensor(data)

    def __repr__(self):
        return f"TorchTensor({self.data})"
    
    # Basic operations
    def __add__(self, other):
        other_TorchTensor = other.data if isinstance(other, TorchTensor) else other
        return TorchTensor(self.data + other_TorchTensor, device=self.device)
    
    def __mul__(self, other):
        other_TorchTensor = other.data if isinstance(other, TorchTensor) else other
        return TorchTensor(self.data * other_TorchTensor, device=self.device)
    
    def __matmul__(self, other):
        """Matrix multiplication using @ operator."""
        other_TorchTensor = other.data if isinstance(other, TorchTensor) else other
        return TorchTensor(torch.matmul(self.data, other_TorchTensor), device=self.device)
    
    # def dot(self, other):
    #     """Alias for matrix multiplication, similar to numpy's dot."""
    #     return self @ other
    
    # Common TorchTensor operations
    def reshape(self, *shape):
        return TorchTensor(self.data.reshape(*shape), device=self.device)
    
    def sum(self, dim=None, keepdim=False):
        return TorchTensor(self.data.sum(dim=dim, keepdim=keepdim), device=self.device)
    
    def mean(self, dim=None, keepdim=False):
        return TorchTensor(self.data.mean(dim=dim, keepdim=keepdim), device=self.device)
    
    # Gradient-related methods
    def backward(self):
        self.data.backward()
    
    def grad(self):
        return TorchTensor(self.data.grad, device=self.device) if self.data.grad is not None else None
    
    # Shape and device properties
    @property
    def shape(self):
        return self.data.shape
    
    # Static methods for TorchTensor creation
    @staticmethod
    def zeros(*shape, dtype=None, device=None):
        return TorchTensor(torch.zeros(*shape, dtype=dtype, device=device), device=device)
    
    @staticmethod
    def ones(*shape, dtype=None, device=None):
        return TorchTensor(torch.ones(*shape, dtype=dtype, device=device), device=device)
    
    @staticmethod
    def randn(*shape, device=None):
        return TorchTensor(torch.randn(*shape, device=device), device=device)
    
    @staticmethod
    def set_seed(seed=3):
        random.seed(seed)
        torch.manual_seed(seed)
    
    # Right-side operations
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        other_TorchTensor = other.data if isinstance(other, TorchTensor) else other
        return TorchTensor(self.data - other_TorchTensor, device=self.device)
    
    def __rsub__(self, other):
        other_TorchTensor = other.data if isinstance(other, TorchTensor) else other
        return TorchTensor(other_TorchTensor - self.data, device=self.device)
    
    def __truediv__(self, other):
        other_TorchTensor = other.data if isinstance(other, TorchTensor) else other
        return TorchTensor(self.data / other_TorchTensor, device=self.device)
    
    def __rtruediv__(self, other):
        other_TorchTensor = other.data if isinstance(other, TorchTensor) else other
        return TorchTensor(other_TorchTensor / self.data, device=self.device)
    
    def exp(self):
        """Returns e raised to the power of each element in the TorchTensor."""
        return TorchTensor(torch.exp(self.data), device=self.device)
    
    def __neg__(self):
        """Returns the negation of the TorchTensor."""
        return TorchTensor(-self.data, device=self.device)

    def argmax(self, dim=None):
        """Returns the indices of the maximum values along a dimension.
        If dim is None, the TorchTensor is flattened before finding the argmax."""
        return TorchTensor(torch.argmax(self.data, dim=dim), device=self.device)

    def transpose(self, dim0=0, dim1=1):
        """Returns a TorchTensor that is a transposed version of the input."""
        return TorchTensor(self.data.transpose(dim0, dim1), device=self.device)
    
    def T(self):
        """Alias for transpose."""
        return self.transpose()
    
    def to(self, device):
        """Move TorchTensor to specified device (e.g. 'cuda' or 'cpu')"""
        if not device:
            print("No device specified, returning original TorchTensor")
            return self
            
        # Check if TorchTensor is already on the requested device
        if self.device == device:
            return self
            
        self.device = device
        return TorchTensor(self.data.to(device), device=device)
    
    # Helper methods
    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def tolist(self):
        """Convert TorchTensor to a Python list."""
        return self.data.cpu().detach().numpy().tolist()
    
    def numpy(self):
        """Convert TorchTensor to a numpy array."""
        return self.data.cpu().detach().numpy()
    
    def unsqueeze(self, dim):
        """Add a dimension of size 1 at the specified position."""
        return TorchTensor(self.data.unsqueeze(dim), device=self.device)

    def squeeze(self, dim=None):
        """Remove dimensions of size 1 from the TorchTensor.
        
        Args:
            dim (int, optional): If given, only squeeze the dimension specified.
                               If None, remove all dimensions of size 1.
        """
        return TorchTensor(self.data.squeeze(dim), device=self.device)

    @staticmethod
    def stack(TorchTensors, dim=0):
        """Stack a sequence of TorchTensors along a new dimension.
        
        Args:
            TorchTensors: sequence of TorchTensors to stack
            dim: dimension to insert. Default: 0
        """
        # Convert any non-TorchTensor elements to TorchTensors first
        TorchTensor_data = [t.data if isinstance(t, TorchTensor) else torch.Tensor(t) for t in TorchTensors]
        # Get the device from the first TorchTensor (if any have one)
        device = next((t.device for t in TorchTensors if isinstance(t, TorchTensor) and t.device), None)
        return TorchTensor(torch.stack(TorchTensor_data, dim=dim), device=device)

    def __pow__(self, other):
        other_TorchTensor = other.data if isinstance(other, TorchTensor) else other
        return TorchTensor(self.data ** other_TorchTensor, device=self.device)
    
    def __rpow__(self, other):
        other_TorchTensor = other.data if isinstance(other, TorchTensor) else other
        return TorchTensor(other_TorchTensor ** self.data, device=self.device)

    def astype(self, dtype):
        """Convert tensor to a different dtype"""
        # Map Python types to torch types
        dtype_map = {
            int: torch.int64,
            float: torch.float32,
            bool: torch.bool
        }
        torch_dtype = dtype_map.get(dtype, dtype)
        return TorchTensor(self.data.to(dtype=torch_dtype), device=self.device)

    @staticmethod
    def ones_like(tensor, dtype=None, device=None):
        """Returns a tensor of ones with the same shape as input tensor"""
        if isinstance(tensor, TorchTensor):
            tensor = tensor.data
        return TorchTensor(torch.ones_like(tensor, dtype=dtype, device=device), device=device)

    def norm(self):
        """Returns the L2 (Euclidean) norm of the tensor."""
        return TorchTensor(torch.norm(self.data), device=self.device)

    def log(self):
        """Returns the natural logarithm of the tensor elements"""
        return TorchTensor(torch.log(self.data), device=self.device)
    
    @staticmethod
    def nan_to_num(tensor, nan=0.0, posinf=None, neginf=None):
        """Replace NaN, positive infinity, and negative infinity with specified values"""
        if isinstance(tensor, TorchTensor):
            tensor = tensor.data
        return TorchTensor(torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf))

    def clip(self, min_val, max_val):
        """Clips tensor values between min_val and max_val"""
        return TorchTensor(torch.clamp(self.data, min_val, max_val), device=self.device)

    def abs(self):
        """Returns the absolute value of each element in the tensor"""
        return TorchTensor(torch.abs(self.data), device=self.device)

class Tensor(TorchTensor if USE_TORCH else NumpyTensor):
    pass
