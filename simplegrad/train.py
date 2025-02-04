import time
import random
from .tensor import Tensor
from .parameter import Parameter
from .evals import evaluate

def SGD(model, training_data, epochs, mini_batch_size, lr,
        test_data=None, test_interval=None, loss_func='mse'):
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

            logits = model(xb)
            loss = getattr(logits, loss_func)(yb)

            model.zero_grad() # Should be optimizer
            loss.backward() # Should be loss.backward() not net.backward(logits)
            for p in model.parameters():
                p.data -= (lr/mini_batch_size) * p.grad.sum(dim=0)

            if test_interval is not None and k % test_interval == 0:
                print(f'[{k*mini_batch_size}/{n}]: {evaluate(model, test_data, batch_size=test_interval)} / {len(test_data)} correct')

        if test_data:
            print(f"Epoch {j}: {evaluate(model, test_data)} / {len(test_data)}, took {time.time()-time1:.2f} seconds")
        else:
            print(f"Epoch {j} complete in {time.time()-time1:.2f} seconds")
