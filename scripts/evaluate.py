
import pickle
import torch
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Dataset class
class SessionDataset(Dataset):
    def __init__(self, sessions, max_len=5):
        self.sessions = sessions
        self.max_len = max_len

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions[idx]
        session = session[-self.max_len :]  # Limit session length
        input_seq = [0] * (self.max_len - len(session)) + session[
            :-1
        ]  # Padding + input sequence
        target = session[-1]  # Target item
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(
            target, dtype=torch.long
        )


# Đọc dữ liệu test
with open("data/processed/test_sessions.pkl", "rb") as f:
    test_sessions = pickle.load(f)

# Tạo dataset và dataloader
max_len = 20
test_dataset = SessionDataset(test_sessions, max_len=max_len)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# Giả lập mô hình
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(max_len, 52739)  # Giả lập số lượng item

    def forward(self, x):
        return self.fc(x.float())


# Hàm negative sampling loss
def negative_sampling_loss(outputs, targets, num_negatives=100, num_items=52739):
    batch_size = targets.size(0)
    positive_logits = outputs[range(batch_size), targets]
    negative_indices = torch.randint(1, num_items + 1, (batch_size, num_negatives))
    negative_logits = outputs.gather(1, negative_indices)
    loss = -torch.mean(
        F.logsigmoid(positive_logits) + torch.sum(F.logsigmoid(-negative_logits), dim=1)
    )
    return loss


# Hàm đánh giá
def evaluate(model, test_dataloader, num_items):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            loss = negative_sampling_loss(
                outputs, targets, num_negatives=100, num_items=num_items
            )
            total_loss += loss.item()
    avg_loss = total_loss / len(test_dataloader)
    return avg_loss


# Load mô hình
model = Model()
model.load_state_dict(torch.load("models/model.pth"))

# Đánh giá
num_items = 52739
avg_test_loss = evaluate(model, test_dataloader, num_items)

# Lưu metrics
with open("metrics/eval_metrics.json", "w") as f:
    json.dump({"test_loss": avg_test_loss}, f)

print(f"Evaluation completed. Test loss: {avg_test_loss}")

