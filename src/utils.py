import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y

def tokenize_text(filepath, process = True):
    with open(filepath, "r", encoding = "utf-8") as file:
        text = file.read().lower()
    if process:
        text = text.lstrip("\ufeff")
        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
    chars = sorted(list(set(text)))
    encoder = {ch: i for i, ch in enumerate(chars)}
    decoder = {i: ch for i, ch in enumerate(chars)}
    encoded = [encoder[ch] for ch in text]
    return text, encoded, encoder, decoder

def prepare_dataloaders(encoded, block_size, batch_size = 16, valid_split = 0.2, seed = 42):
    dataset = CharDataset(torch.tensor(encoded, dtype = torch.long), block_size)
    n = len(dataset)
    valid_len = int(n * valid_split)
    train_len = n - valid_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, valid_ds = random_split(dataset, [train_len, valid_len], generator = generator)
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(valid_ds, batch_size = batch_size, shuffle = False)
    return train_loader, valid_loader

class TransformerBlock(nn.Module):
    def __init__(self, dim, dropout = 0.1):
        super().__init__()
        self.dim = dim
        self.W_q = nn.Linear(dim, dim) # What features is this token looking for?
        self.W_k = nn.Linear(dim, dim) # What features does this token contain?
        self.W_v = nn.Linear(dim, dim) # What features does this token pass along?
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, 4 * dim)
        self.ff2 = nn.Linear(4 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        x_norm = self.attn_norm(x)
        Q = self.W_q(x_norm)
        K = self.W_k(x_norm)
        V = self.W_v(x_norm)
        scores = Q @ K.transpose(-1, -2) / (self.dim ** (1 / 2))
        mask = torch.tril(torch.ones(T, T, device = x.device)) # Prevents tokens from looking at future tokens
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim = -1)
        out = self.dropout(attn_weights @ V)
        x = x + out
        x_norm = self.ff_norm(x)
        hidden = torch.relu(self.ff1(x_norm))
        mlp_out = self.dropout(self.ff2(hidden))
        x = x + mlp_out
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, dim, n_transformers):
        super().__init__()
        self.block_size = block_size
        self.d_model = dim
        self.token_embedding = nn.Embedding(vocab_size, dim) # vocab_size x dim
        self.position_embedding = nn.Embedding(block_size, dim) # block_size x dim
        self.blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(n_transformers)])
        self.linear = nn.Linear(dim, vocab_size)

    def forward(self, batch):
        _, T = batch.shape
        if T > self.block_size:
            raise ValueError(f"Input sequence length T = {T} exceeds block_size = {self.block_size}")
        token_emb = self.token_embedding(batch) # batch_size x block_size x dim
        pos_emb = self.position_embedding(torch.arange(T, device = batch.device))[None, :, :] # 1 x block_size x dim
        x = token_emb + pos_emb # batch_size x block_size x dim
        for block in self.blocks:
            x = block(x)
        logits = self.linear(x) # batch_size x block_size x vocab_size
        return logits
    
def train_model(model, train_loader, valid_loader, lr, min_epochs = 100, max_epochs = None, patience = 50):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    train_loss_history = []
    valid_loss_history = []
    best_valid_loss = float("inf")
    epochs_since_improvement = 0
    epoch = 0
    while True:
        epoch += 1
        print(f"Current epoch: {epoch}", end = "\r", flush = True)
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y.view(B * T))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                B, T, V = logits.shape
                loss = criterion(logits.view(B * T, V), y.view(B * T))
                valid_loss += loss.item() * len(x)
        valid_loss /= len(valid_loader.dataset)
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        if epoch >= min_epochs:
            if valid_loss < best_valid_loss - 1e-4:
                best_valid_loss = valid_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    break
        if max_epochs is not None and epoch >= max_epochs:
            break
    history = {"train": train_loss_history, "valid": valid_loss_history}
    return model, history

def forecast(model, text, horizon, encoder, decoder, block_size, temperature = None, top_k = None):
    model.eval()
    device = next(model.parameters()).device
    x = [encoder[ch] for ch in text.lower()]
    for _ in range(horizon):
        context = x[-block_size:] if len(x) >= block_size else x
        context_tensor = torch.tensor(context, dtype = torch.long, device = device).unsqueeze(0)
        with torch.no_grad():
            logits = model(context_tensor)
        logits = logits[0, -1]
        if temperature is None:
            next_id = torch.argmax(logits).item()
        else:
            logits = logits / temperature  # temperature < 1: sharpens the distribution, > 1: flattens the distribution
            if top_k is not None:  # Forces the model to only consider top k choices
                values, indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask[indices] = values
                logits = mask
            probs = F.softmax(logits, dim = -1)
            next_id = torch.multinomial(probs, num_samples = 1).item()
        x.append(next_id)
    return "".join(decoder[i] for i in x)


def forecast_sentence(model, text, encoder, decoder, block_size, temperature = None, top_k = None, cutoff = 300):
    model.eval()
    device = next(model.parameters()).device
    x = [encoder[ch] for ch in text.lower()]
    steps = 0
    while True:
        if cutoff is not None and steps >= cutoff:
            break
        context = x[-block_size:] if len(x) >= block_size else x
        context_tensor = torch.tensor(context, dtype = torch.long, device = device).unsqueeze(0)
        with torch.no_grad():
            logits = model(context_tensor)
        logits = logits[0, -1]
        if temperature is None:
            next_id = torch.argmax(logits).item()
        else:
            logits = logits / temperature
            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask[indices] = values
                logits = mask
            probs = F.softmax(logits, dim = -1)
            next_id = torch.multinomial(probs, num_samples = 1).item()
        x.append(next_id)
        ch = decoder[next_id]
        steps += 1
        if ch in ".?!":
            break
    return "".join(decoder[i] for i in x)


def plot_loss_history(history):
    plt.figure(figsize = (10, 5))
    plt.plot(history["train"], label = "Train Loss")
    plt.plot(history["valid"], label = "Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()