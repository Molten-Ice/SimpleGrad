from .tensor import Tensor

def evaluate(model, test_data, batch_size=128):
    if not test_data:
        return 'No test data provided'
    model.eval()
    correct = 0
    for i in range(0, len(test_data), batch_size):

        batch = test_data[i:i + batch_size]
        x = Tensor.stack([x for x, _ in batch])
        y = Tensor.stack([y for _, y in batch]).squeeze(dim=1).tolist()
        
        outputs = model(x)
        predictions = outputs.data.argmax(dim=1).squeeze(dim=1).tolist()
        correct += sum(int(pred == label) for pred, label in zip(predictions, y))
        
    model.train()
    return correct