from .tensor import Tensor

def evaluate(net, test_data, batch_size=128):
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