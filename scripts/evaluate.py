# evaluate.py
import torch
from .train import negative_sampling_loss

def evaluate(model, test_dataloader, num_items):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = negative_sampling_loss(outputs, targets, num_negatives=100, num_items=num_items)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_dataloader)
    return avg_loss