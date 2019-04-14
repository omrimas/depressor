import os
import json
import re
import voc
import unicodedata
import pickle
import numpy as np
import bcolz

words = []
idx = 0
word2idx = {}
glove_path = 'data/glove'
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/840B.300.dat', mode='w')

with open(f'{glove_path}/glove.840B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        if not re.match('^[a-zA-Z0-9_]+$', word):
            continue

        try:
            vect = np.array(line[1:]).astype(np.float)
        except ValueError:
            continue
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vectors.append(vect)
        if idx % 100 == 0:
            print(idx)

vectors = bcolz.carray(vectors[1:].reshape((-1, 300)), rootdir=f'{glove_path}/840B.300.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/840B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/840B.300_idx.pkl', 'wb'))