import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from pathlib import Path

# Đảm bảo thư mục đầu ra tồn tại
Path("models").mkdir(parents=True, exist_ok=True)
Path("metrics").mkdir(parents=True, exist_ok=True)


# Dataset class
class SessionDataset(Dataset):
    def __init__(self, sessions, max_len=20):
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
train_sessions_file = "data/processed/train_sessions.pkl"
test_sessions_file = "data/processed/test_sessions.pkl"

if not os.path.exists(train_sessions_file):
    raise FileNotFoundError(
        f"Train sessions file {train_sessions_file} does not exist."
    )
if not os.path.exists(test_sessions_file):
    raise FileNotFoundError(f"Test sessions file {test_sessions_file} does not exist.")

try:
    with open(train_sessions_file, "rb") as f:
        sessions = pickle.load(f)
    with open(test_sessions_file, "rb") as f:
        test_sessions = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error reading session files: {str(e)}")

# Tạo dataset và dataloader
max_len = 20
dataset = SessionDataset(sessions, max_len=max_len)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
print(f"Train dataset created with {len(dataset)} sessions")

test_dataset = SessionDataset(test_sessions, max_len=max_len)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print(f"Test dataset created with {len(test_dataset)} sessions")


# SASRec model
class SASRec(nn.Module):
    def __init__(
        self,
        num_items,
        embedding_dim,
        hidden_size,
        num_heads,
        num_layers,
        dropout=0.1,
        max_len=100,
    ):
        super(SASRec, self).__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_encoding = self._generate_pos_encoding(embedding_dim, max_len)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(embedding_dim, num_items + 1)

    def _generate_pos_encoding(self, embedding_dim, max_len):
        pos = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-np.log(10000.0) / embedding_dim)
        )
        pos_encoding = torch.zeros(max_len, embedding_dim)
        pos_encoding[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding[:, 1::2] = torch.cos(pos * div_term)
        return pos_encoding.unsqueeze(0)  # Không chuyển lên cuda ngay


    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :].to(x.device)
        mask = (x.sum(dim=-1) == 0).bool()  # Mask for padding
        output = self.transformer(x, src_key_padding_mask=mask)
        output = self.fc(output[:, -1, :])  # Predict next item
        return output


# Hàm negative sampling loss
def negative_sampling_loss(outputs, targets, num_negatives=100, num_items=52739):
    batch_size = targets.size(0)
    positive_logits = outputs[range(batch_size), targets]

    # Tạo negative indices, đảm bảo không trùng với targets
    negative_indices = torch.zeros(
        batch_size, num_negatives, dtype=torch.long, device=targets.device
    )
    for i in range(batch_size):
        neg_candidates = torch.arange(1, num_items + 1, device=targets.device)
        mask = neg_candidates != targets[i]
        neg_candidates = neg_candidates[mask]
        negative_indices[i] = neg_candidates[
            torch.randperm(len(neg_candidates))[:num_negatives]
        ]

    negative_logits = outputs.gather(1, negative_indices)
    loss = -torch.mean(
        F.logsigmoid(positive_logits) + torch.sum(F.logsigmoid(-negative_logits), dim=1)
    )
    return loss


# Hàm đánh giá
def evaluate(model, test_dataloader, num_items, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = negative_sampling_loss(
                outputs, targets, num_negatives=100, num_items=num_items
            )
            total_loss += loss.item()
    avg_loss = total_loss / len(test_dataloader)
    return avg_loss


# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_items = 52739  # Số item duy nhất từ preprocess.py
model = SASRec(
    num_items=num_items,
    embedding_dim=128,
    hidden_size=256,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
    max_len=max_len,
).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 2

mlflow.set_tracking_uri("http://192.168.28.39:5000")
mlflow.set_experiment("YooChoose_SASRec_Tracking")

with mlflow.start_run(run_name="SASRec_NegativeSampling"):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = negative_sampling_loss(outputs, targets, num_items=num_items)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log loss cho mỗi batch
            mlflow.log_metric(
                f"batch_train_loss_epoch_{epoch}",
                loss.item(),
                step=epoch * len(dataloader) + batch_idx,
            )

        avg_train_loss = total_loss / len(dataloader)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        # Evaluate
        avg_test_loss = evaluate(model, test_dataloader, num_items, device)
        mlflow.log_metric("test_loss", avg_test_loss, step=epoch)

    # Lưu mô hình
    try:
        torch.save(model.state_dict(), "models/model.pth")
        mlflow.pytorch.log_model(model, "final_model")
    except Exception as e:
        raise RuntimeError(f"Error saving model: {str(e)}")

    # Lưu metrics
    try:
        with open("metrics/metrics.json", "w") as f:
            import json

            json.dump({"train_loss": avg_train_loss, "test_loss": avg_test_loss}, f)
    except Exception as e:
        raise RuntimeError(f"Error saving metrics.json: {str(e)}")

