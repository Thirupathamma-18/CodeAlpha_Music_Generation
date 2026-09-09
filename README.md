# AI Music Generation using LSTM

## CodeAlpha Internship – Task 3

An AI-based music generation system that uses an LSTM neural network to learn musical patterns and generate new music sequences.

## Objective
To generate new music using Deep Learning and sequential pattern learning.

## Technologies Used
- Python
- PyTorch
- Music21
- NumPy
- LSTM
- MIDI

## Workflow
Dataset → Preprocessing → Note & Chord Extraction → Numerical Encoding → LSTM Training → Music Generation → MIDI Output

## Dataset
- 50 music files from the Music21 corpus
- 7,689 musical elements extracted
- 66 unique musical elements
- Sequence length: 50

## Model
The LSTM model consists of an Embedding Layer, 2 LSTM Layers, and a Fully Connected Layer.

Training Parameters:
- Epochs: 20
- Batch Size: 64
- Hidden Size: 128
- Learning Rate: 0.001
- Optimizer: Adam

Training loss decreased from 3.3960 to 0.6127.

## Result
The trained model successfully generated 200 musical elements and converted them into a playable MIDI file.

Output: output/generated_music.mid

## Project Structure
CodeAlpha_Music_Generation/
├── dataset/
├── output/
│   └── generated_music.mid
├── collect_midi.py
├── preprocess.py
├── train_model.py
├── generate_music.py
├── notes.pkl
├── music_lstm.pth
├── requirements.txt
└── README.md

## How to Run
python preprocess.py
python train_model.py
python generate_music.py

The generated music will be saved in output/generated_music.mid.

## Future Scope
- Larger music datasets
- Longer compositions
- Multiple instruments and genres
- Web-based music generation

## Conclusion
This project demonstrates how LSTM and Deep Learning can be used to learn musical patterns and generate new playable music.

## Author
CodeAlpha Internship – Task 3
Project: AI Music Generation using LSTM
