import pickle
import torch
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from torch.optim import Adam
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


# Đọc sessions
with open("data/processed/train_sessions.pkl", "rb") as f:
    sessions = pickle.load(f)
with open("data/processed/test_sessions.pkl", "rb") as f:
    test_sessions = pickle.load(f)

# Tạo dataset và dataloader
max_len = 20
dataset = SessionDataset(sessions, max_len=max_len)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
print(f"Train dataset created with {len(dataset)} sessions")

test_dataset = SessionDataset(test_sessions, max_len=max_len)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print(f"Test dataset created with {len(test_dataset)} sessions")


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


# Training
model = Model()
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 2
num_items = 52739

mlflow.set_tracking_uri("http://192.168.28.39:5000")
mlflow.set_experiment("YooChoose_SASRec_Tracking")

with mlflow.start_run(run_name="SASRec_NegativeSampling"):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = negative_sampling_loss(outputs, targets, num_items=num_items)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(dataloader)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        # Evaluate
        avg_test_loss = evaluate(model, test_dataloader, num_items)
        mlflow.log_metric("test_loss", avg_test_loss, step=epoch)

    # Lưu mô hình
    torch.save(model.state_dict(), "models/model.pth")
    mlflow.pytorch.log_model(model, "final_model")

    # Lưu metrics
    with open("metrics/metrics.json", "w") as f:
        import json

        json.dump({"train_loss": avg_train_loss, "test_loss": avg_test_loss}, f)
