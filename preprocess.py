from music21 import converter, instrument, note, chord
from pathlib import Path
import pickle

dataset_dir = Path("dataset")

music_files = list(dataset_dir.glob("*.mxl")) + \
              list(dataset_dir.glob("*.xml")) + \
              list(dataset_dir.glob("*.musicxml"))

print("Music files found:", len(music_files))

# Store musical symbols
notes = []

for file_path in music_files:
    print("Processing:", file_path.name)

    try:
        music = converter.parse(str(file_path))

        parts = instrument.partitionByInstrument(music)

        if parts:
            elements = parts.parts[0].recurse()
        else:
            elements = music.flatten().notes

        for element in elements:

            # Single note
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            # Chord
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    except Exception as e:
        print("Error processing:", file_path.name)
        print(e)

print()
print("Total musical elements:", len(notes))

# Create vocabulary
pitchnames = sorted(set(notes))

print("Unique musical elements:", len(pitchnames))

# Convert musical symbols to numbers
note_to_int = {note_name: number for number, note_name in enumerate(pitchnames)}

numerical_sequence = [
    note_to_int[note_name]
    for note_name in notes
]

# Save processed data
data = {
    "notes": notes,
    "pitchnames": pitchnames,
    "note_to_int": note_to_int,
    "numerical_sequence": numerical_sequence
}

with open("notes.pkl", "wb") as f:
    pickle.dump(data, f)

print()
print("Preprocessing completed successfully!")
print("Saved file: notes.pkl")
print("Numerical sequence length:", len(numerical_sequence))