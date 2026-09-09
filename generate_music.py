import pickle
import torch
import torch.nn as nn
from music21 import stream, note, chord, tempo
from pathlib import Path
import random

# Load preprocessed data
with open("notes.pkl", "rb") as f:
    data = pickle.load(f)

pitchnames = data["pitchnames"]

# Load trained model
checkpoint = torch.load(
    "music_lstm.pth",
    map_location="cpu"
)

vocab_size = checkpoint["vocab_size"]
sequence_length = checkpoint["sequence_length"]
hidden_size = checkpoint["hidden_size"]
num_layers = checkpoint["num_layers"]


# LSTM model
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
model = MusicLSTM(vocab_size)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Trained model loaded successfully!")


# Starting sequence
start_index = random.randint(
    0,
    len(data["numerical_sequence"]) - sequence_length
)

pattern = data["numerical_sequence"][
    start_index:start_index + sequence_length
]

generated = []

# Generate 200 musical elements
for _ in range(200):

    input_sequence = torch.tensor(
        [pattern],
        dtype=torch.long
    )

    with torch.no_grad():

        prediction = model(input_sequence)

        probabilities = torch.softmax(
            prediction / 1.0,
            dim=1
        )

        next_index = torch.multinomial(
            probabilities,
            1
        ).item()

    generated.append(
        pitchnames[next_index]
    )

    pattern.append(next_index)
    pattern = pattern[1:]


print("Generated musical elements:", len(generated))


# Convert generated sequence to MIDI
music = stream.Stream()

# Add tempo
music.append(
    tempo.MetronomeMark(number=100)
)

for element in generated:

    try:

        # Chord
        if "." in element:

            pitches = [
                int(p)
                for p in element.split(".")
            ]

            new_chord = chord.Chord(pitches)
            new_chord.quarterLength = 0.5

            music.append(new_chord)

        # Note
        else:

            new_note = note.Note(element)
            new_note.quarterLength = 0.5

            music.append(new_note)

    except Exception:
        pass


# Create output folder
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "generated_music.mid"

music.write(
    "midi",
    fp=str(output_file)
)

print()
print("🎵 Music generation completed!")
print("MIDI file saved at:")
print(output_file)