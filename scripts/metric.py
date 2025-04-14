# metric.py
import torch

def recall_at_20(model, dataloader, num_items, k=20):
    model.eval()
    total_recall = 0
    num_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, top_k = torch.topk(outputs, k, dim=-1)
            matches = (top_k == targets.unsqueeze(-1)).sum(dim=-1)
            total_recall += matches.sum().item()
            num_samples += targets.numel()
    return total_recall / num_samples

def mrr_at_k(model, dataloader, num_items, k=20):
    model.eval()
    total_mrr = 0
    num_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, top_k = torch.topk(outputs, k, dim=-1)
            for i, target in enumerate(targets):
                rank = (top_k[i] == target).nonzero(as_tuple=True)[0]
                if rank.numel() > 0:
                    total_mrr += 1.0 / (rank.item() + 1)
            num_samples += targets.size(0)
    return total_mrr / num_samples