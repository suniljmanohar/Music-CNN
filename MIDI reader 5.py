from mido import MidiFile
from MIDI_functions import draw_music, convert_to_pitch_classes, convert_midi_to_grid
from MIDI_functions import steps_per_beat, stride, label_length_steps, folder, inputs_fname, steps_per_sample
import os
import time
import numpy as np

# Locations
training_subfolder = "training_files"  # training files in this subfolder
test_subfolder = "test_files"  # test files in this subfolder

visualise_samples = False

prog_start = time.time()
output = {}
folder_paths = {"train": folder + training_subfolder + "/",
                "test": folder + test_subfolder + "/"}

for key in folder_paths.keys():
    input_folder = folder_paths[key]

    sample_collection, label_collection = [], []  # list of downsized samples and labels for import into tensorflow
    PCs_sample_collection, PCs_label_collection = [], []  # similar for pitchclasses

    # read in raw data from all files
    for fname in os.listdir(input_folder):  # iterate over all files in target folder
        if fname[-3:] == "mid":
            file_start = time.time()
            m = MidiFile(os.path.join(input_folder, fname))

            # initialise some variables for converting timer from ticks to steps
            ticks_per_beat = m.ticks_per_beat
            ticks_per_step = ticks_per_beat//steps_per_beat
            print("Processing " + fname + " (ticks per beat: " + str(ticks_per_beat) + ")")

            music_grid, no_offs = convert_midi_to_grid(m, ticks_per_step)
            if no_offs:
                print(fname + " has no off-note messages! Not usable for learning.")
                continue

            # cut out samples and labels
            for sample_start in range(0, len(music_grid) - steps_per_sample - label_length_steps, stride):
                sample_end = sample_start + steps_per_sample

                # cut out sample and add to collection
                sample = music_grid[sample_start:sample_end, :]
                sample_collection.append(sample)
                PCs_sample_collection.append(convert_to_pitch_classes(sample))

                # cut out subsequent label and add to collection
                label = music_grid[sample_end:sample_end + label_length_steps, :]
                label_collection.append(label)
                PCs_label_collection.append(convert_to_pitch_classes(label))

                # draw sample "musically"
                if visualise_samples:
                    draw_music(sample, steps_per_sample)
            file_stop = time.time()
            # print("   {0:.2f}".format(file_stop - file_start) + " seconds")

    output[key + '_features'] = np.asarray(sample_collection).astype(np.int8)
    output[key + '_labels'] = np.asarray(label_collection).astype(np.int8)
    output[key + '_PCs_features'] = np.asarray(PCs_sample_collection).astype(np.int8)
    output[key + '_PCs_labels'] = np.asarray(PCs_label_collection).astype(np.int8)

# export data
results = np.asarray([output['train_features'], output['train_labels'], output['test_features'], output['test_labels']])
results_PCs = np.asarray([output['train_PCs_features'], output['train_PCs_labels'], output['test_PCs_features'], output['test_PCs_labels']])
np.save(folder + inputs_fname, results)
np.save(folder + inputs_fname + "_PCs.npy", results_PCs)

# display counts
print("Total number of samples processed: " + str(results[0].shape[0]) + " for training and "+ str(results[2].shape[0]) + " for testing")
print("Total number of labels processed: " + str(results[1].shape[0]) + " for training and " + str(results[3].shape[0]) + " for testing")
prog_end = time.time()
print("Time taken: " + "{0:.2f}".format(prog_end-prog_start) + "s")


