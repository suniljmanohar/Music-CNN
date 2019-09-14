from MIDI_functions import draw_music, probs_to_chord, convert_grid_to_midi
from MIDI_functions import steps_per_beat, beats_per_sample, folder, predictions_fname, inputs_fname
import numpy as np

# Filenames
output_folder = "Results MIDI files/"
visualise_samples = False

# Import data
features = np.load(folder + inputs_fname, allow_pickle=True)[0]  # index 0 to load training samples, 2 for test
predictions = np.load(folder + predictions_fname, allow_pickle=True)[0]  # index 0 to load training preds, 1 for test
labels = np.load(folder + inputs_fname, allow_pickle=True)[1]

try:
    assert len(predictions) == len(features)
except AssertionError:
    print("Predictions and features do not match.")

print("Samples to process: " + str(len(features)))

# iterate over all samples
for i in range(len(features)):
    new_chord = probs_to_chord(features[i, -1], predictions[i], steps_per_beat)
    note_array = np.concatenate((features[i], new_chord))
    if visualise_samples:
        draw_music(note_array, steps_per_beat * beats_per_sample)
    m = convert_grid_to_midi(note_array, steps_per_beat, 500000)
    m.save(folder + output_folder + "test_output" + str(i) + ".mid")
    if i % 1000 == 0: print(str(i) + " samples processed")
