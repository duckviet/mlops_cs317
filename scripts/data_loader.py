# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader

class SessionDataset(Dataset):
    def __init__(self, sessions, max_len=5):
        self.sessions = sessions
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session = self.sessions[idx]
        session = session[-self.max_len:]  # Limit session length
        input_seq = [0] * (self.max_len - len(session)) + session[:-1]  # Padding + input sequence
        target = session[-1]  # Target item
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def get_data_loaders(sessions, test_sessions, max_len=20, batch_size=128):
    dataset = SessionDataset(sessions, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = SessionDataset(test_sessions, max_len=max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train dataset created with {len(dataset)} sessions")
    print(f"Test dataset created with {len(test_dataset)} sessions")

    return dataloader, test_dataloader