import time
import random
from .tensor import Tensor
from .parameter import Parameter
from .evals import evaluate

def SGD(model, training_data, epochs, mini_batch_size, lr,
        test_data=None, test_interval=None, loss_func='mse',
        lambda_reg = 0.1):
    model.eval()
    print(f"Initial evaluation: {evaluate(model, test_data)} / {len(test_data)}")

    model.train()
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
            # Note: doesn't include regularization loss (which is directly adjusted in backprop in (scale_factor * p.data) below.
            loss = getattr(logits, loss_func)(yb)

            model.zero_grad() # Should be optimizer
            loss.backward() # Should be loss.backward() not net.backward(logits)

            scale_factor = 1- lr*lambda_reg/mini_batch_size
            for p in model.parameters():
                if p.is_weight:
                    p.data = (scale_factor * p.data) - lr * p.grad.sum(dim=0)
                else:
                    p.data -= lr * p.grad.sum(dim=0)

            if test_interval is not None and k % test_interval == 0:
                model.eval()
                print(f'[{k*mini_batch_size}/{n}]: {evaluate(model, test_data, batch_size=test_interval)} / {len(test_data)} correct')
                model.train()

        if test_data:
            model.eval()
            print(f"Epoch {j}: {evaluate(model, test_data)} / {len(test_data)}, took {time.time()-time1:.2f} seconds")
            model.train()
        else:
            print(f"Epoch {j} complete in {time.time()-time1:.2f} seconds")

