# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import time
from .data_loader import get_data_loaders
from .preprocess import load_and_preprocess_data

# SASRec model
class SASRec(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_size, num_heads, num_layers, dropout=0.1):
        super(SASRec, self).__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_encoding = self._generate_pos_encoding(embedding_dim, max_len=100)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_size, 
                dropout=dropout, batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embedding_dim, num_items + 1)
    
    def _generate_pos_encoding(self, embedding_dim, max_len):
        pos = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pos_encoding = torch.zeros(max_len, embedding_dim)
        pos_encoding[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding[:, 1::2] = torch.cos(pos * div_term)
        return pos_encoding.unsqueeze(0).cuda()
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :].to(x.device)
        mask = (x.sum(dim=-1) == 0).bool()  # Mask for padding
        output = self.transformer(x, src_key_padding_mask=mask)
        output = self.fc(output[:, -1, :])  # Predict next item
        return output

# Loss function
def negative_sampling_loss(outputs, targets, num_negatives=100, num_items=52739):
    batch_size = targets.size(0)
    positive_logits = outputs[range(batch_size), targets]
    negative_indices = torch.randint(1, num_items + 1, (batch_size, num_negatives)).cuda()
    negative_logits = outputs.gather(1, negative_indices)
    loss = -torch.mean(F.logsigmoid(positive_logits) + torch.sum(F.logsigmoid(-negative_logits), dim=1))
    return loss

def train_model(file_path, num_epochs=2, learning_rate=0.001, num_negatives=100, patience=2):
    # Load and preprocess data
    sessions, test_sessions, num_items = load_and_preprocess_data(file_path)

    # Get data loaders
    dataloader, test_dataloader = get_data_loaders(sessions, test_sessions, max_len=20, batch_size=128)

    # Initialize model
    model = SASRec(
        num_items=num_items,
        embedding_dim=64,
        hidden_size=128,
        num_heads=4,
        num_layers=2
    ).cuda()

    # Set up training
    mlflow.set_tracking_uri("http://192.168.28.39:5000")
    mlflow.set_experiment("YooChoose_SASRec_Tracking")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint_path = "best_model.pth"

    with mlflow.start_run(run_name="SASRec_NegativeSampling"):
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("num_items", num_items)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_negatives", num_negatives)
        mlflow.log_param("patience", patience)
        mlflow.log_param("batch_size", 128)
        mlflow.log_param("max_len", 20)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            start_time = time.time()

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = negative_sampling_loss(outputs, targets, num_negatives=num_negatives, num_items=num_items)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

            avg_train_loss = total_loss / len(dataloader)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f} seconds")

            # Evaluate
            avg_test_loss = evaluate(model, test_dataloader, num_items)
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Test Loss: {avg_test_loss:.4f}")

            # Log metrics
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("test_loss", avg_test_loss, step=epoch)
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)

            # Early stopping
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                patience_counter = 0
                torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict(), 'loss': best_loss}, 
                           checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                print(f"Saved best model at Epoch {epoch+1} with Test Loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered after Epoch {epoch+1}. Best Test Loss: {best_loss:.4f}")
                break

        mlflow.pytorch.log_model(model, "final_model")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded best model from {checkpoint_path} with Test Loss: {checkpoint['loss']:.4f}")

    return model

if __name__ == "__main__":
    file_path = '/home/nguyenvietduc-22520273/.rs_datasets'
    train_model(file_path)