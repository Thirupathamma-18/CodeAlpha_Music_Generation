from music21 import corpus
from pathlib import Path
import shutil

dataset_dir = Path("dataset")
dataset_dir.mkdir(exist_ok=True)

all_paths = corpus.getPaths()

# Use music21 files that can be converted to MIDI
music_files = [
    Path(p) for p in all_paths
    if Path(p).suffix.lower() in [".mxl", ".xml", ".musicxml"]
]

print("Music files available:", len(music_files))

# Copy first 50 files
count = 0

for source in music_files[:50]:
    destination = dataset_dir / source.name

    if destination.exists():
        destination = dataset_dir / f"{count}_{source.name}"

    try:
        shutil.copy2(source, destination)
        count += 1
        print("Copied:", source.name)

    except Exception as e:
        print("Error:", source.name, e)

print()
print(f"Music files copied to dataset: {count}")