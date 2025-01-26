import time
import math
import random
from tensor import Tensor
from parameter import Parameter

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
        # print(f'[w @ x + b]: {self.w.data.shape} @ {x.data.shape} + {self.b.data.shape}')
        z = self.w @ x + self.b
        # print(f'z.data.shape: {z.data.shape}')
        a = z.sigmoid()
        return a
    
    # [w @ x + b]: (10, 784) @ (3, 784, 1) + (10, 1)
    # ---------- __matmul__ ----------
    # (10, 784) @ (3, 784, 1)
    # out.data.shape: (3, 10, 1)
    # z.data.shape: (3, 10, 1)
    # --------------------------------------------------
    # [w @ x + b]: (10, 10) @ (3, 10, 1) + (10, 1)
    # ---------- __matmul__ ----------
    # (10, 10) @ (3, 10, 1)
    # out.data.shape: (3, 10, 1)
    # z.data.shape: (3, 10, 1)
    
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

    
def backward(self, x):
    for l in range(2, len(self.layers)+1):
        # x1. Backpropagate to previous layer
        if l == 2: # Base case.
            loss_function = self.layers[-1]
            grad = loss_function.metadata['derivative']
        else:
            next_layer = self.layers[-l+1]
            grad = next_layer.w.transpose(-2, -1) @ grad # Shape remains unchanged.

        # x2. Backpropagate through activation function
        layer = self.layers[-l]
        grad *= layer.activation_function.derivative(layer.metadata['z'])

        # x3a. Backpropagate to bias (leaf node)
        layer.metadata['nabla_b'] += grad.sum(dim=0) # sum(dim=0) squashes batches.

        # x3b. Backpropagate to weights (leaf node)
        activation = x if l == len(self.layers) else self.layers[-l-1].metadata['a']
        layer.metadata['nabla_w'] += (grad @  activation.transpose(-2, -1)).sum(dim=0)

    # w @ x
    # grad = next_layer.w.transpose(-2, -1) @ grad # Shape remains unchanged.
    # layer.metadata['nabla_w'] += grad @ activation.transpose(-2, -1)


    import sys; sys.exit()

def SGD(model, training_data, epochs, mini_batch_size, eta,
        test_data=None, test_interval=None):
    print(f"Initial evaluation: {evaluate(model, test_data)} / {len(test_data)}")


    n = len(training_data)
    for j in range(epochs):
        time1 = time.time()
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in range(0, n, mini_batch_size)]
        for k, mini_batch in enumerate(mini_batches):
            # print('*'*50, f'mini_batch {k}', '*'*50)
            xb, yb = map(lambda t: Tensor.stack(t, dim=0), zip(*mini_batch))
            xb, yb = Parameter(xb, _op='xb'), Parameter(yb, _op='yb')  # Add this line
            
            mini_batch_size = xb.data.shape[0]
            lr = eta/len(mini_batch)

            logits = model(xb)
            loss = logits.mse(yb)
            # print(f"Loss: {loss.data:.5f}")

            model.zero_grad() # Should be optimizer
            loss.backward() # Should be loss.backward() not net.backward(logits)
            for p in model.parameters():
                p.data -= lr * p.grad.sum(dim=0)

            if test_interval is not None and k % test_interval == 0:
                print(f'[{k*mini_batch_size}/{n}]: {evaluate(model, test_data, batch_size=test_interval)} / {len(test_data)} correct')

        if test_data:
            print(f"Epoch {j}: {evaluate(model, test_data)} / {len(test_data)}, took {time.time()-time1:.2f} seconds")
        else:
            print(f"Epoch {j} complete in {time.time()-time1:.2f} seconds")



def evaluate(net, test_data, batch_size=3): # 128
    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation.
    
    Args:
        test_data: List of (x, y) tuples containing test inputs and labels
        batch_size: Size of batches to process at once (default: 32)
    """
    
    correct = 0
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        x = Tensor.stack([x for x, _ in batch])
        y = Tensor.stack([y for _, y in batch]).squeeze(dim=1).tolist()
        
        outputs = net(x)
        predictions = outputs.data.argmax(dim=1).squeeze(dim=1).tolist()
        correct += sum(int(pred == label) for pred, label in zip(predictions, y))
        
    return correct