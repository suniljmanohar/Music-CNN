import numpy as np, random as rnd
import matplotlib.pyplot as plt
from mido import merge_tracks, MidiFile, MidiTrack, Message, MetaMessage
from tensorflow import keras

beats_per_sample = 8  # in beats
steps_per_beat = 12  # resolution to downsize samples to
stride = 2  # number of STEPS between start of each sample
label_length_steps = 1  # length of target segment, currently set to single chord
folder = "e:/programming/PyCharmProjects/MIDI reader/"  # place subfolders in here
predictions_fname = "predictions.npy"
inputs_fname = "MIDI_data_input.npy"
checkpoint_dir = "checkpoints/"
steps_per_sample = beats_per_sample*steps_per_beat


class ConvModel(keras.Sequential):
    def __init__(self, steps_per_sample):
        super().__init__()
        self.add(keras.layers.Conv2D(64, kernel_size=(15, 15), activation='relu', input_shape=(steps_per_sample, 128, 1)))
        self.add(keras.layers.Conv2D(128, (5, 5), activation='relu'))
        self.add(keras.layers.MaxPooling2D(pool_size=(4, 1)))
        self.add(keras.layers.Conv2D(128, (5, 5), activation='relu'))
        self.add(keras.layers.MaxPooling2D(pool_size=(4, 1)))
        self.add(keras.layers.Dropout(0.25))
        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(128, activation='relu'))
        self.add(keras.layers.Dropout(0.5))
        self.add(keras.layers.Dense(128, activation='softmax'))


# 2-stave visualisation. 'graph' is 2D array of (pitches, times), 'width' is the scaling of the time axis
def draw_music(grid, width):
    graph = np.where(grid == 1)
    bass_tuples = [(x,y) for y, x in zip(graph[0], graph[1]) if x < 60]
    xbass = [x[1] for x in bass_tuples]
    ybass = [x[0] for x in bass_tuples]
    treble_tuples = [(x, y) for y, x in zip(graph[0], graph[1]) if x > 59]
    xtreble = [x[1] for x in treble_tuples]
    ytreble = [x[0] for x in treble_tuples]
    fig, (ax1,ax2) = plt.subplots(2)
    ax1.set_xlim([0, width])
    ax2.set_xlim([0, width])
    ax1.set_ylim([49, 84])
    ax2.set_ylim([30, 62])
    ax1.scatter(xtreble, ytreble)
    for l in [64, 67, 71, 74, 77]:
        ax1.axhline(y=l)
    ax2.scatter(xbass, ybass)
    for l in [57, 53, 50, 47, 44]:
        ax2.axhline(y=l)
    ax1.axis('off')
    ax2.axis('off')
    plt.show()
    return


def convert_to_pitch_classes(sample):
    find_ones = np.where(sample == 1)
    converted_sample = np.zeros((len(sample), 12))
    converted_sample[find_ones[0], find_ones[1] % 12] = 1
    return converted_sample


def convert_midi_to_grid(m, ticks_per_step):  # m is a MidiFile object
    # merge all tracks into one
    trklist = []
    for trk in m.tracks:
        trklist.append(trk)
    trk = merge_tracks(trklist)

    raw_notes = [[0] * 128]  # 2D array of (128 pitches, time in ticks), entries are 1 for note on, 0 for note off
    timer = 0  # measured in steps, continues through the entire MIDI file
    no_offs = True
    # iterate through all midi messages in file and create grid of raw music
    for msg in trk:
        delta_t = msg.time  # in ticks
        delta_steps = (timer + delta_t)//ticks_per_step - timer//ticks_per_step
        if delta_steps > 0:  # fill in array until next MIDI event then advance timer
            raw_notes += [raw_notes[timer//ticks_per_step][:]] * (delta_steps - 1)
            raw_notes += [raw_notes[timer//ticks_per_step][:]]

        timer += delta_t

        # NB fast notes will be clustered!
        if msg.type == 'note_on':
            raw_notes[timer//ticks_per_step][msg.note] = 1
        elif msg.type == 'note_off':
            raw_notes[timer//ticks_per_step][msg.note] = 0
            no_offs = False

    return np.asarray(raw_notes), no_offs


def convert_grid_to_midi(grid, ticks_per_beat, tempo):
    # create blank MIDI file and track
    m = MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    trk = MidiTrack()
    m.tracks.append(trk)
    trk.append(MetaMessage('set_tempo', tempo=tempo))
    # add padding zeros at beginning and end of sample to allow calculation of on and off messages
    grid = np.concatenate((np.zeros((1, 128)), grid, np.zeros((1, 128))))

    t_last_event = 0
    # iterate over sample and compare notes on/off at time t and t+1
    for t in range(grid.shape[0]-1):
        for pitch in range(grid.shape[1]):
            change = grid[t+1, pitch] - grid[t, pitch]
            if change == 1:
                trk.append(Message('note_on', note=pitch, time=(t - t_last_event), velocity=100))
                t_last_event = t

            if change == -1:
                trk.append(Message('note_off', note=pitch, time=(t - t_last_event)))
                t_last_event = t
    return m


def probs_to_chord(prev_chord, probs, length):  # picks the most likely
    n_notes = int(np.sum(prev_chord))  # count number of notes "on" in preceding chord
    likeliest_notes = np.argpartition(probs, -n_notes)[-n_notes:]  # pick n notes with highest probabilities
    predicted_chord = np.zeros(128)
    predicted_chord[likeliest_notes] = 1
    predicted_chord = (np.ones((length, 1)) * predicted_chord)
    return predicted_chord


class NoteSample:
    def __init__(self, l):
        self.pitches = 128
        self.length = l
        self.notes = np.zeros((self.length, self.pitches))

    def transpose(self, shift):  # transposes whole sample by shift semitones
        print("Transposed by " + str(shift))
        result = np.zeros_like(self.notes)
        for t in range(self.length):
            for p in range(max(0, -shift), min(self.pitches, self.pitches - shift)):
                result[t, p + shift] = self.notes[t, p]
        return result

    def stretch(self, scale):  # stretches time dimension of sample by scale
        print("Stretched by " + str(scale))
        result = []
        sample = list(self.notes)
        for t in range(len(sample)):
            result += [sample[t]] * scale
        return np.asarray(result)

    def drop_note(self):  # drops a random note from sample
        result = np.zeros_like(self.notes)
        ons = self.find_ons(self.notes)
        note = ons[rnd.randint(0, len(ons) - 1)]
        print("Dropped note " + str(note))
        t, p = note[0], note[1]
        while t < self.length and sample[t, p] == 1:
            result[t, p] = 0
            t += 1
        return result

    def find_top(self):
        result = np.zeros_like(self.notes)
        for t in range(self.length):
            p = self.pitches - 1
            while self.notes[t, p] == 0 and p >= 0:
                p -= 1
            if p >= 0:
                result[t, p] = 1
        return result

    def find_bass(self):
        result = np.zeros_like(self.notes)
        for t in range(self.length):
            p = 0
            while p < self.pitches and self.notes[t, p] == 0:
                p += 1
            if p < self.pitches:
                result[t, p] = 1
        return result

    def add_octave(self):
        return self.notes + self.transpose(self.find_bass(), -12)

    def find_ons(self, sample):
        note_ons = []
        prev_chord = np.zeros(sample.shape[1], dtype=np.int8)
        for t in range(sample.shape[0]):
            for pitch in range(sample.shape[1]):
                change = sample[t, pitch] - prev_chord[pitch]
                if change == 1:
                    note_ons += [[t, pitch]]
            prev_chord = sample[t]
        return note_ons
