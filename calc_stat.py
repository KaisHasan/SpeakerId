import os
import numpy as np
import gc
import tensorflow as tf
from tqdm import tqdm
from scipy.io import wavfile
import pre_processing as pr


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

DIR = 'Datasets/Voxceleb'
m = len(sorted(os.listdir(DIR)))
SIZE = 5000  # Number of examples to load into memory at once
LIMIT = 100_000  # Max number of examples to be loaded
START = 0
MAX_LENGTH = 48_000  # The max length of the audio segment (3s)

# A permutation to load the examples in a shuffled way
perm = np.load('vox_perm.npy')

files = sorted(os.listdir(DIR))
# The sum of the mean value and standard deviation for chunck of examples
means, stds = 0, 0
row = 0
for it in range(1):
    print(f'*****epoch: {it}*****')
    st, en = START, START + SIZE
    c = 0
    while st < LIMIT + START:
        en = min(en, LIMIT + START)
        print(f'\t{st} -> {en}')
        data = []
        # load the data
        for k in tqdm(range(st, en)):
            i = perm[k]

            f = files[i]
            sample_rate, wav_file = wavfile.read(DIR + '/' + f)
            if wav_file.size > MAX_LENGTH:
                wav_file = wav_file[:MAX_LENGTH]
            data.append(wav_file)
        gc.collect()
        print('\t\tpadding the data')
        data = pr.padding(data)
        gc.collect()
        print('\t\tconvert to spectrogram')
        data = pr.convert_to_spectrogram(data, sample_rate)
        gc.collect()
        print('\t\tnormalize')

        means += np.mean(data)
        stds += np.std(data)

        st += SIZE
        en += SIZE
        del data
        gc.collect()

# Here we divide by the numbers of chunck
# We had loaded 1e5 examples and each chunck have 5e3
# So we had to divide by 1e5/5e3 = 20
means /= 20
stds /= 20

print(f'means: {means}')
print(f'stds: {stds}')
