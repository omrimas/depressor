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


def load_glove(voc, weights_matrix):
    print("adding glove words")
    with open(GLOVE_FILE, 'r', encoding="utf8") as f:
        for idx, line in enumerate(f):
            if idx % 50000 == 0:
                print("Finished %s words" % idx)
            items = line.strip().split()
            word, vec = items[0], [float(x) for x in items[1:]]
            voc.add_word(word)  # add to vocabulary
            weights_matrix[idx + 1] = vec  # add to weights matrix


def add_training(voc, weights_matrix):
    print("adding train words")
    not_in_glove = 0
    with open(TRAIN_FILE, 'r') as f:
        for idx, line in enumerate(f):
            if idx % 50000 == 0:
                print("Finished %s lines" % idx)

            # add to weights matrix
            for w in set(line.split()):
                if w not in voc.word2index:
                    weights_matrix[GLOVE_VEC_CNT + not_in_glove] = np.random.normal(scale=0.6, size=(300,))
                    not_in_glove += 1

            voc.add_sentence(line)

        print("words not in glove: %d" % not_in_glove)
        print(weights_matrix)


def build_annoy(voc, weights_matrix):
    print("build annoy index")
    annoy_obj = annoy.AnnoyIndex(GLOVE_VEC_LEN)
    for idx, _ in enumerate(voc.word2index):
        if idx % 5000 == 0:
            print("Finished %s words" % idx)
        annoy_obj.add_item(idx, weights_matrix[idx])
    annoy_obj.build(10)
    annoy_obj.save(ANNOY_FILE)


if __name__ == '__main__':
    vocab = voc.Voc("depression")
    weights_matrix = np.zeros((GLOVE_VEC_CNT + WORDS_NOT_IN_GLOVE, GLOVE_VEC_LEN))

    load_glove(vocab, weights_matrix)
    add_training(vocab, weights_matrix)

    print("Saving weights matrix")
    f_wm = open(WEIGHTS_MATRIX_PICKLE, "wb")
    pickle.dump(weights_matrix, f_wm, protocol=4)
    f_wm.close()

    print("Saving vocabulary")
    f_voc = open(VOC_PICKLE, "wb")
    pickle.dump(vocab, f_voc, protocol=4)
    f_voc.close()

    build_annoy(vocab, weights_matrix)
