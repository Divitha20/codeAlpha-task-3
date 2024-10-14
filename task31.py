import music21

def sequence_to_midi(sequence, output_path='generated_sample_music.mid'):
    """Convert a sequence of note events back to a MIDI file."""
    stream = music21.stream.Stream()
    
    for note in sequence:
        if note == 'Rest':
            stream.append(music21.note.Rest())
        else:
            stream.append(music21.note.Note(note))
    
    stream.write('midi', fp=output_path)

# Sample sequence of notes (you can replace this with generated notes)
generated_notes = [2, 4, 5, 10, 8, 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest', 'Rest']

# Convert the sequence to a MIDI file
sequence_to_midi(generated_notes, 'generated_sample_music.mid')
print("MIDI file created successfully!")