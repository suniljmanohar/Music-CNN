# Music-CNN
Convolutional neural network for predicting classical music

This is an hobby project which uses Tensorflow to apply a convolutional neural network to pieces of classical music (currently testing on Bach). MIDI files are used as the source data; each file is converted to a "time-against-pitch" grid of 1s and 0s. The output tensor os the network is a "chord vector" of 128 pitches which is compared to the actual next chord in the piece. A loss function and gradient descent are used to minimise the difference between the predicted chord and actual chord.

Files:

1. MIDI_functions. Contains various functions for manipulating MIDI files and converting them to and from grids. Also contains the structure of the network model
2. MIDI reader. Contains code to convert a batch of test and training MIDI files into arrays of "features" and "labels"
3. MIDI CNN. Builds and trains the model, and creates a set of prediction chords for each sample
4. MIDI writer. Used to append the predicted chords to the original samples and convert them to MIDI, to listen to the output
5. training_files and test_files. MIDI files for use with MIDI reader. Currently contains various works by JS Bach
