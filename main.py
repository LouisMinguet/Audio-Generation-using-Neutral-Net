!pip install mido==1.2.9
!pip install music21==6.7.1

import pandas as pd
import numpy as np
from mido import MidiFile
import IPython
import matplotlib.pyplot as plt
import librosa.display
import keras.layers as L
import keras.models as M
import keras
from keras.layers import SimpleRNN,LSTM,GRU
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, LSTM, Conv2DTranspose, Conv2D, LeakyReLU, GlobalMaxPooling2D, Reshape, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers.embeddings import Embedding
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam

from sklearn.model_selection import train_test_split
from IPython import *
import os
import tensorflow as tf

from numpy.random import choice

from mido import Message, MidiFile, MidiTrack

key_notes = {
    'Cb': 59,
    'C': 60,
    'C#': 61,
    'Db': 61,
    'D': 62,
    'D#': 63,
    'Eb': 63,
    'E': 64,
    'F': 65,
    'F#': 66,
    'Gb': 66,
    'G': 67,
    'G#': 68,
    'Ab': 68,
    'A': 69,
    'A#': 70,
    'Bb': 70,
    'B': 71
}

data=pd.read_csv('../input/musicnet-dataset/musicnet_metadata.csv')

data.head()

mid=MidiFile('../input/musicnet-dataset/musicnet_midis/musicnet_midis/Beethoven/2313_qt15_1.mid',clip=True)
mid.tracks

for i in mid.tracks[1] :
    if 'meta' in str(i):
        print(i)

mid.tracks[1].name

k = 0
for i in mid.tracks[1] :
    print(i)
    k += 1
    if k > 50:
        break

beethoven_midi_traks = {}
n=10
name = None
for m in range(n):
    mid=MidiFile('../input/musicnet-dataset/musicnet_midis/musicnet_midis/Beethoven/'+os.listdir('../input/musicnet-dataset/musicnet_midis/musicnet_midis/Beethoven')[m],clip=True)
    print(mid.tracks)
    for j in range(len(mid.tracks)):
        if j == 0:
            name = mid.tracks[j].name + ': '
            print(name)
        else:
            beethoven_midi_traks[name + mid.tracks[j].name] = mid.tracks[j]

beethoven_midi_traks

def get_key(s):
    k = None
    if 'key' in s:
        k = s[33:35]
        if k[-1] == "m" or k[-1] == "'":
            k = k[:-1]
    return k


# function returning list of dicts with each note, it's duration and velocity 
def parse_notes(track):
    key = 'C'
    tunes = []
    new_tune = []
    note_dict = {}
    for i in track:
        
        if i.is_meta:
            new_key = get_key(str(i))
            if new_key is not None:
                key = new_key
            if len(tunes) > 0:
                tunes.append(new_tune)
                new_tune = []
                
        elif i.type == 'note_on' or i.type == 'note_off':
            if i.type == 'note_on' and i.dict()['velocity'] > 0 and i.dict()['time'] > 0:
                note_dict['time'] = i.dict()['time']
                note_dict['note'] = i.dict()['note']
                note_dict['velocity'] = i.dict()['velocity']
                note_dict['channel'] = i.dict()['channel']
            elif i.type == 'note_off' or i.type == 'note_on' and i.dict()['velocity'] == 0:
                if note_dict:
                    note_dict['pause'] = i.dict()['time']
                    note_dict['key'] = key
                    new_tune.append(note_dict)
                    note_dict = {}
    tunes.append(new_tune)
    return tunes

def tune_to_midi(tune, midi_name='new_tune', debug_mode=False):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for note in tune:
        if debug_mode:
            track.append(Message('note_on', note=note, time=64))
            track.append(Message('note_off', note=note, time=128))
        else:
            track.append(Message('note_on', note=note['note'], velocity=note['velocity'], time=note['time']))
            track.append(Message('note_off', note=note['note'], time=note['pause']))

    mid.save(midi_name + '.mid')

    tunes = []
max_key = max(key_notes.values())
for k, v in beethoven_midi_traks.items():
    if 'Right' in k:
        new_tunes = parse_notes(v)
        if len(new_tunes) > 0:
            tunes.append(pd.DataFrame(new_tunes[0]))

tunes[0]

def various(notes):
    flag = True
    for i in range(8, len(notes)):
        flag = len(np.unique(notes[i-8:i])) > 2
        if not flag:
            break
    return flag

phrase_len = 60
X = []
y = []
for t in tunes:
    for i in range(len(t) - phrase_len):
        if various(t.iloc[i:i + phrase_len, 1]):
            X.append(t.iloc[i:i + phrase_len, :3])
            y.append(t.iloc[i + phrase_len, :3])
X = np.array(X)
y = np.array(y)

X = X.astype(int)
y = y.astype(int)

X.shape

# LSTM 

model = Sequential()
model.add(LSTM(512,return_sequences=False, input_shape=(phrase_len, 3)))
model.add(Dropout(0.5))
model.add(Dense(3, activation='relu'))
model.compile(loss='mae', optimizer='adam')

model.fit(X, y, batch_size=256, epochs=70, validation_split=0.2)

history=model.history.history
plt.plot([i for i in range(len(history['loss']))],history['loss'])
plt.plot([i for i in range(len(history['val_loss']))],history['val_loss'])

def tune_generator(model, name='lstm_tune_'):
    for i in range(3):
        start = np.random.randint(0, len(X)-1)
        pattern = X[start]
        prediction_output = []

        for note_index in range(100):
            prediction_input = np.reshape(pattern, (1, len(pattern), 3))
            prediction = model.predict(prediction_input, verbose=0)
            prediction_output.append(prediction.astype(int)[0])
            pattern = np.append(pattern, prediction, axis = 0)
            pattern = pattern[1:len(pattern)]

        notes = pd.DataFrame(prediction_output, columns=['time', 'note', 'velocity'])
        notes['pause'] = 180
        notes_dict = notes.to_dict('records')
        tune_to_midi(notes_dict, midi_name=name + str(i))

tune_generator(model, name='lstm_mod_')

# LSTM using embedding

n_notes = 128
embed_size = 100

notes_in = Input(shape = (phrase_len,))
durations_in = Input(shape = (phrase_len,1))

notes_embed = Embedding(n_notes, embed_size)(notes_in)

concat_model = Concatenate()([notes_embed,durations_in])
concat_model = Dropout(0.3)(concat_model)
concat_model = LSTM(512, return_sequences=False)(concat_model)

notes_out = Dense(n_notes, activation = 'softmax', name = 'note')(concat_model)
durations_out = Dense(1, activation = 'relu', name = 'duration')(concat_model)

embed_model = Model([notes_in, durations_in], [notes_out, durations_out])
embed_model.compile(loss=['sparse_categorical_crossentropy', 
                    'mse'], optimizer=RMSprop(lr = 0.001))

train_chords = X[:, :, 1]
train_durations = X[:, :, 0]
target_chords = y[:, 1]
target_durations = y[:, 0]

embed_model.fit([train_chords, train_durations], 
                    [target_chords, target_durations]
                    , epochs=200, batch_size=256, validation_split=0.2
                  )

history=embed_model.history.history
plt.plot([i for i in range(len(history['note_loss']))],history['note_loss'])
plt.plot([i for i in range(len(history['val_note_loss']))],history['val_note_loss'])

for i in range(3):
    start = np.random.randint(0, len(X)-1)
    pattern_chords = X[start, :, 1]
    pattern_durations = X[start, :, 0]

    prediction_output = []
    
    for note_index in range(100):
        pattern_chords = np.reshape(pattern_chords, (1, len(pattern_chords), 1))
        pattern_durations = np.reshape(pattern_durations, (1, len(pattern_durations), 1))

        prediction = embed_model.predict([pattern_chords, pattern_durations], verbose=0)
        index = np.random.choice(n_notes, p=prediction[0][0])
        duration = prediction[1][0][0]

        prediction_output.append([index, int(duration)])

        pattern_chords = np.append(pattern_chords, index)
        pattern_chords = pattern_chords[1:len(pattern_chords)]

        pattern_durations = np.append(pattern_durations, duration)
        pattern_durations = pattern_durations[1:len(pattern_durations)]

    notes = pd.DataFrame(prediction_output, columns=['note', 'time'])
    notes['pause'] = 180
    notes['velocity'] = 80
    notes_dict = notes.to_dict('records')
    tune_to_midi(notes_dict, midi_name='embed_lstm_' + str(i))

# GAN

u, c = np.unique(X[:, :, 0].sum(axis=1), return_counts=True)
np.median(u)

def tune_to_matrix(tune, tune_len=4000):
    notes_matrix = np.zeros((128, tune_len))
    i = 0
    for n in tune:
        for j in range(int(n[0] / 4)):
            notes_matrix[n[1], i] = 1
            i += 1
            if i == tune_len:
                break
        if i == tune_len:
            break
    return notes_matrix

X[0, :, 0].sum()

tune_to_matrix(X[0]).sum(axis=1)

tune_len = 200
n_notes = 128

train_matrixes = []
for x in X[:1000]:
    train_matrixes.append(tune_to_matrix(x, tune_len=200))

len(train_matrixes)

# Fitting the model

latent_dim = 64

discriminator = Sequential(
    [
        Input((n_notes, tune_len)),
        Reshape((n_notes, tune_len, 1)),
        Conv2D(4, (8, 4), padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2D(8, (8, 4), padding="same"),
        LeakyReLU(alpha=0.2),
        GlobalMaxPooling2D(),
        Dense(1, activation='sigmoid'),
    ],
    name="discriminator",
)

generator = Sequential(
    [
        Input((latent_dim,)),
        Dense(n_notes * tune_len * latent_dim),
        Reshape((n_notes, tune_len, latent_dim)),
        Conv2DTranspose(1, (4, 2), padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(1, (4, 2), padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2D(1, (6, 4)),
        Flatten(),
        Dense(n_notes * tune_len, activation='sigmoid'),
        Reshape((n_notes, tune_len)),
    ],
    name="generator",
)

generator.summary()

class GAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.distr = tf.random.uniform

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, real_tunes):
        # Sample random points in the latent space 
        # This is for the generator.
        batch_size = tf.shape(real_tunes)[0]
        random_latent_vectors = self.distr(shape=(batch_size, self.latent_dim))


        # Decode the noise (guided by labels) to fake images.
        generated_tunes= self.generator(random_latent_vectors)
        
        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions_real = self.discriminator(real_tunes)
            predictions_fake = self.discriminator(generated_tunes)
            d_loss = self.d_loss_fn([predictions_fake, predictions_real], [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))])
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = self.distr(shape=(batch_size, self.latent_dim))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_tunes = self.generator(random_latent_vectors)
            predictions = self.discriminator(fake_tunes)
            g_loss = self.g_loss_fn(predictions, tf.zeros((batch_size, 1)))
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

dataset = np.array(train_matrixes[:256])

def discriminator_loss(pred, labels):
    real_loss = tf.reduce_mean(pred[1])
    fake_loss = tf.reduce_mean(pred[0])
    return real_loss - 2 * fake_loss


def generator_loss(pred, labels):
    return tf.reduce_mean(pred)

gan = GAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.000005),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    d_loss_fn=discriminator_loss,
    g_loss_fn=generator_loss
)

gan.fit(dataset, epochs=60)

history=gan.history.history
plt.plot([i for i in range(len(history['g_loss']))],history['g_loss'])

plt.plot([i for i in range(len(history['d_loss']))],history['d_loss'])

def extract_tune(matrix):
    time = 0
    note = -1
    tune = []
    for i in range(matrix.shape[1]):
        new_note = np.argmax(matrix[:, i])
        if note != new_note:
            if note != -1:
                tune.append([time, note])
            note = new_note
            time = 4
        else:
            time += 4
    tune.append([time, note])
    return tune

for i in range(3):
    random_latent_vectors = tf.random.uniform(shape=(1, latent_dim))
    tunes = gan.generator(random_latent_vectors)
    t = extract_tune(tunes[0])
    notes = pd.DataFrame(t, columns=['time', 'note'])
    notes['pause'] = 180
    notes['velocity'] = 80
    notes_dict = notes.to_dict('records')
    tune_to_midi(notes_dict, midi_name='gan_mod_' + str(i))