import torch
import annoy
import os
import voc
import numpy as np
import pickle

TRAIN_FILE = os.path.join("data", "fixed_training")
GLOVE_FILE = os.path.join("data", "glove.6B.300d.txt")
GLOVE_VEC_LEN = 300
GLOVE_VEC_CNT = 400002
WORDS_NOT_IN_GLOVE = 88862
ANNOY_FILE = GLOVE_FILE + ".annoy"
WEIGHTS_MATRIX_PICKLE = os.path.join("pickles", "weights.matrix.pkl")
VOC_PICKLE = os.path.join("pickles", "voc.pkl")
MIN_COUNT_WORDS = 4


def load_glove(voc, weights_matrix):
    with open(GLOVE_FILE, 'r', encoding="utf8") as f:
        for idx, line in enumerate(f):
            if idx % 50000 == 0:
                print("Finished %s words" % idx)
            items = line.strip().split()
            word, vec = items[0], [float(x) for x in items[1:]]
            if word in voc.word2index:
                voc.glove_words.add(word)
                weights_matrix[voc.word2index[word]] = vec  # add to weights matrix


def add_training(voc):
    with open(TRAIN_FILE, 'r') as f:
        for idx, line in enumerate(f):
            if idx % 50000 == 0:
                print("Finished %s lines" % idx)

            voc.add_sentence(line)


def complete_weight_mat(voc, weights_matrix):
    for idx, w in enumerate(voc.word2index):
        if idx % 5000 == 0:
            print("Finished %s words" % idx)

        if w == '<PAD>':
            weights_matrix[voc.word2index[w]] = np.zeros(shape=(300,))
        else:
            if w not in voc.glove_words:
                weights_matrix[voc.word2index[w]] = np.random.normal(scale=0.6, size=(300,))


def build_annoy(voc, weights_matrix):
    annoy_obj = annoy.AnnoyIndex(GLOVE_VEC_LEN)
    for idx, word in enumerate(voc.word2index):
        if idx % 5000 == 0:
            print("Finished %s words" % idx)
        if word in voc.glove_words:
            if word == '<PAD>':
                continue
            annoy_obj.add_item(idx, weights_matrix[idx])
    annoy_obj.build(100)
    annoy_obj.save(ANNOY_FILE)


if __name__ == '__main__':
    vocab = voc.Voc("depression")

    print("adding train words to vocabulary...")
    add_training(vocab)
    print("Done!")
    print("%d words in vocabulary" % vocab.num_words)

    print("Trimming words with less than %d occurrences" % MIN_COUNT_WORDS)
    vocab.trim(MIN_COUNT_WORDS)
    print("%d words in vocabulary after trimming" % vocab.num_words)

    print("Marking glove words in vocabulary and adding their vectors to weights matrix...")
    weights_matrix = np.zeros((vocab.num_words, GLOVE_VEC_LEN))
    load_glove(vocab, weights_matrix)
    print("Done!")

    print("Completing weights matrix for non-glove words...")
    complete_weight_mat(vocab, weights_matrix)
    print("Done!")

    print("Saving weights matrix")
    f_wm = open(WEIGHTS_MATRIX_PICKLE, "wb")
    pickle.dump(weights_matrix, f_wm, protocol=4)
    f_wm.close()
    print("Done!")

    print("Saving vocabulary")
    f_voc = open(VOC_PICKLE, "wb")
    pickle.dump(vocab, f_voc, protocol=4)
    f_voc.close()
    print("Done!")

    print("Building annoy index...")
    build_annoy(vocab, weights_matrix)
    print("Done!")
