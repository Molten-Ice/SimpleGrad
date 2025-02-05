import time
import random
from .tensor import Tensor
from .parameter import Parameter
from .evals import evaluate


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def step(self):
        for p in self.parameters():  # Call parameters() each time
            if p.grad is not None:
                p.data -= self.lr * p.grad.sum(dim=0)


def SGD(model, training_data, epochs, batch_size, lr,
        test_data=[], test_interval=None, loss_func='mse',
        lambda_reg = 0):
    model.eval()
    print(f"Initial evaluation: {evaluate(model, test_data)} / {len(test_data)}")

    x_all, y_all = map(lambda t: Tensor.stack(t, dim=0), zip(*training_data))

    optimizer = Optimizer(model.parameters, lr) # Note model.parameters is a bound method, not a list.

    model.train()
    n = x_all.shape[0]
    n_batches = (n + batch_size - 1) // batch_size

    for j in range(epochs):
        time1 = time.time()
        indices = list(range(n))
        random.shuffle(indices)
        for k in range(n_batches):
            batch_indices = indices[k*batch_size:(k+1)*batch_size]

            xb, yb = x_all[batch_indices], y_all[batch_indices]
            xb, yb = Parameter(xb, _op='xb'), Parameter(yb, _op='yb')  # Add this line
            current_batch_size = xb.data.shape[0]

            optimizer.zero_grad()
            output = model(xb)

            loss = getattr(output, loss_func)(yb)
            if lambda_reg > 0:
                base_loss, l2_loss = loss, calculate_l2_loss(model, current_batch_size, lambda_reg)
                loss += l2_loss


            loss.backward()
            optimizer.step()

            if test_interval is not None and k % test_interval == 0:
                weight_sizes = [round(p.data.abs().mean().data.item(), 5) for p in model.parameters() if p.is_weight]
                extra = f' -> base_loss: {base_loss.data.data:.5f}' + (f', l2_loss: {l2_loss.data.data:.5f}' if lambda_reg > 0 else '') + f', weight_sizes: {weight_sizes}'
                print(f'[{k*batch_size}/{n}]: {evaluate(model, test_data, batch_size=test_interval)} / {len(test_data)} correct{extra}')

        print(f"Epoch {j}: {evaluate(model, test_data)} / {len(test_data)}, took {time.time()-time1:.2f} seconds")
   
def calculate_l2_loss(model, mini_batch_size, lambda_reg):
    l2_loss = Parameter(Tensor(0.0))
    for p in model.parameters():
        if p.is_weight:
            l2_loss = l2_loss + (lambda_reg / (2 * mini_batch_size)) * (p * p).sum()
    return l2_loss
