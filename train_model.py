import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load preprocessed data
with open("notes.pkl", "rb") as f:
    data = pickle.load(f)

sequence = data["numerical_sequence"]

print("Total sequence length:", len(sequence))

# Parameters
sequence_length = 50
batch_size = 64
hidden_size = 128
num_layers = 2
epochs = 20
learning_rate = 0.001

# Create training sequences
X = []
y = []

for i in range(len(sequence) - sequence_length):
    X.append(sequence[i:i + sequence_length])
    y.append(sequence[i + sequence_length])

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

print("Input shape:", X.shape)
print("Target shape:", y.shape)

# Dataset
class MusicDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


dataset = MusicDataset(X, y)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# LSTM Model
class MusicLSTM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            128
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        output, _ = self.lstm(x)

        output = output[:, -1, :]

        output = self.fc(output)

        return output


# Create model
vocab_size = len(data["pitchnames"])

model = MusicLSTM(vocab_size)

print()
print("Vocabulary size:", vocab_size)
print("Model created successfully!")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# Training
print()
print("Starting training...")

for epoch in range(epochs):

    total_loss = 0

    for inputs, targets in dataloader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f}"
    )


# Save model
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "num_layers": num_layers
    },
    "music_lstm.pth"
)

print()
print("Training completed successfully!")
print("Model saved as: music_lstm.pth")