import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import music21

# Sample note data for testing
notes = [
    'C4', 'D4', 'E4', 'Rest', 'G4', 'A4', 'B4', 'C5', 'Rest',
    'C4', 'E4', 'G4', 'Rest', 'A4', 'C5', 'Rest', 'B4', 'G4', 
    'C4', 'Rest', 'D4', 'F4', 'A4', 'C5', 'E5', 'Rest', 'G5'
]


def prepare_sequences(notes, seq_length):
    """Convert notes to input/output sequences for training."""
    note_to_int = {note: num for num, note in enumerate(sorted(set(notes)))}
    inputs, outputs = [], []
    
    for i in range(len(notes) - seq_length):
        inputs.append([note_to_int[note] for note in notes[i:i + seq_length]])
        outputs.append(note_to_int[notes[i + seq_length]])
    
    X = np.reshape(inputs, (len(inputs), seq_length, 1))  # Reshape for LSTM
    y = np.eye(len(note_to_int))[outputs]  # One-hot encode the outputs
    
    return X, y, note_to_int

# Define the sequence length
sequence_length = 5  # You can adjust this for smaller sequences

# Prepare sequences using the synthetic note data
X, y, note_to_int = prepare_sequences(notes, sequence_length)

# Check the shape of the data
print(f"Input shape: {X.shape}, Output shape: {y.shape}")

# Build the RNN model
input_shape = (X.shape[1], X.shape[2])
model = Sequential()
model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(note_to_int), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model (adjust the epochs and batch size for real training)
model.fit(X, y, epochs=10, batch_size=4)

# Save the model
model.save('sample_music_rnn_model.keras')


def generate_music_rnn(model, seed_sequence, note_to_int, int_to_note, length=50):
    """Generate music using the trained RNN model."""
    generated_notes = seed_sequence.copy()
    
    for _ in range(length):
        X_seed = np.reshape(seed_sequence, (1, len(seed_sequence), 1))
        prediction = model.predict(X_seed, verbose=0)
        next_note = np.argmax(prediction)
        
        seed_sequence.append(next_note)
        seed_sequence = seed_sequence[1:]  # Move to the next sequence
        generated_notes.append(int_to_note[next_note])
    
    return generated_notes

# Convert integer to note dictionary
int_to_note = {num: note for note, num in note_to_int.items()}

# Seed sequence for generating music (use the first few notes as the seed)
seed_sequence = [note_to_int[note] for note in notes[:sequence_length]]

# Generate new music (generate 20 new notes for testing)
generated_notes = generate_music_rnn(model, seed_sequence, note_to_int, int_to_note, length=20)

# Print the generated notes
print("Generated Notes:", generated_notes)


def sequence_to_midi(sequence, output_path='generated_sample_music.mid'):
    """Convert a sequence of note events back to a MIDI file."""
    stream = music21.stream.Stream()
    
    for note in sequence:
        if note == 'Rest':
            stream.append(music21.note.Rest())
        else:
            stream.append(music21.note.Note(note))
    
    stream.write('midi', fp=output_path)

# Convert generated notes to MIDI
sequence_to_midi(generated_notes)