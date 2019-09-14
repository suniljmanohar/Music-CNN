from __future__ import print_function
from tensorflow import keras
from MIDI_functions import ConvModel, steps_per_sample, folder, predictions_fname, inputs_fname, checkpoint_dir
import numpy as np, random as rnd
import matplotlib.pyplot as plt


def regulariser(sample):
    f = rnd.random()
    if f < 0.2:
        return sample.transpose(rnd.randint(0, 12))
    elif f >= 0.2 and f < 0.4:
        return sample.stretch(rnd.randint(1, 8))
    elif f >= 0.4 and f < 0.6:
        return sample.drop_note()
    elif f >= 0.6 and f < 0.8:
        return sample.add_octave()
    else:
        return sample


# Hyperparameters
learn_rate = 0.1
dropout_rate = 0.4
batch_size = 100
epochs = 50
checkpoint_period = 5  # save checkpoints every n training epochs

# import data
data = np.load(folder + inputs_fname, allow_pickle=True)
print(data[0].shape)
x_train = data[0].astype(np.float16)
y_train = data[1].astype(np.float16).reshape(len(data[1]), 128)
x_test = data[2].astype(np.float16)
y_test = data[3].astype(np.float16).reshape(len(data[3]), 128)

# adjust shape to Conv2D requirements
x_train = x_train.reshape(x_train.shape[0], steps_per_sample, 128, 1)
x_test = x_test.reshape(x_test.shape[0],  steps_per_sample, 128, 1)
input_shape = (steps_per_sample, 128, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Create checkpoint callbacks
checkpoint_path = folder + checkpoint_dir + "cp-{epoch:04d}.ckpt"
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1,
                                              period=checkpoint_period)

# Build model
model = ConvModel(steps_per_sample)
model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[cp_callback])
model.save(folder + 'my_model.h5')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate model on test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Create a predictions vector for each sample
pred_train = model.predict_proba(x_train)
pred_test = model.predict_proba(x_test)
np.save(folder + predictions_fname, np.array([pred_train, pred_test]))
