import os
import json
import re
import voc
import unicodedata
import pickle
import numpy as np
import bcolz
from scipy.spatial import distance
import random

GLOVE_PATH = os.path.join("data", "glove")
TRAIN_DATA_PATH = os.path.join("data", "training")
VOCS_PICKLE = os.path.join("pickles", "vocs.pkl")
P = 0.2
MIN_COUNT_WORD = 3
FIXED_TRAINING = os.path.join("data", "fixed_training")
LABEL = "depression"


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'^https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
    s = re.sub(r"\<.*?\>", " ", s)
    s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def build_voc(label):
    vocab = voc.Voc(label)

    none_label = 0
    with open(TRAIN_DATA_PATH, 'r') as f:
        for idx, line in enumerate(f):
            if idx % 100 == 0:
                print("Finished %s lines" % idx)
            line_json = json.loads(line)
            posts, cur_label = line_json[0]["posts"], line_json[0]["label"]
            if cur_label is None:
                none_label += 1
                continue
            if cur_label != label:
                continue
            for _, post in posts:
                normalized_line = normalize_string(post)
                vocab.add_sentence(normalized_line)

    print("none label: ")
    print(none_label)
    return vocab


def filter_data(label):
    lines = []
    with open(TRAIN_DATA_PATH, 'r') as f:
        for idx, line in enumerate(f):
            if idx % 100 == 0:
                print("Finished %s lines" % idx)
            line_json = json.loads(line)
            posts, cur_label = line_json[0]["posts"], line_json[0]["label"]
            if cur_label != label:
                continue
            for _, post in posts:
                normalized_line = normalize_string(post)
                keep_line = True
                for w in normalized_line.split():
                    if w not in voc.word2index:
                        keep_line = False

                if keep_line:
                    lines.append(normalized_line)

    return lines


def build_glove_matrix(glove, voc):
    weights_matrix = np.zeros((voc.num_words, 50))
    words_found = 0
    for i, word in voc.index2word.items():
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(50,))
    print("Words found %d out of %d" % (words_found, voc.num_words))
    return weights_matrix


def get_sim_words_indices(w, weights_matrix):
    word_idx = voc.word2index[w]
    sim_word_idx = word_idx
    if random.uniform(0, 1) < P:
        word_vector = weights_matrix[word_idx]
        distances = distance.cdist([word_vector], weights_matrix, "cosine")[0]
        top_ten = np.argsort(distances)[1:11]
        sim_word_idx = top_ten[random.randint(1, 10)]
    return sim_word_idx


def build_sim_words_dict(voc, weights_matrix):
    index2sim = {}
    for k, v in voc.index2word.items():
        if k % 50 == 0:
            print("Finished %s words" % k)
        word_vector = weights_matrix[k]
        distances = distance.cdist([word_vector], weights_matrix, "cosine")[0]
        index2sim[k] = np.argsort(distances)[random.randint(0, 9)]
    return index2sim


def build_training_data(voc, weights_matrix):
    src = []
    sim_src = []
    with open(FIXED_TRAINING, 'r') as f:
        for idx, line in enumerate(f):
            if (idx % 10 == 0) and (idx > 0):
                print("Finished %s lines" % idx)

            words = line.split()
            if len(words) > 1:
                reg_sentence = [voc.word2index[w] for w in words]
                sim_reg_sentence = [get_sim_words_indices(w, weights_matrix) for w in words]
                src += reg_sentence
                sim_src += sim_reg_sentence

        return src, sim_src


# build vocs
# voc = build_voc("depression")
# voc.trim(MIN_COUNT_WORD)
# f = open(VOCS_PICKLE, "wb")
# pickle.dump(voc, f)
# f.close()

# # filter sentences with trimmed words
# f = open(VOCS_PICKLE, "rb")
# voc = pickle.load(f)
# lines = filter_data("depression")
# with open(FIXED_TRAINING, 'w+') as outf:
#     for line in lines:
#         outf.write(line + '\n')
# outf.close()


# # build glove matrices
# f = open(VOCS_PICKLE, "rb")
# voc = pickle.load(f)
# vectors = bcolz.open(f'{GLOVE_PATH}/6B.50.dat')[:]
# words = pickle.load(open(f'{GLOVE_PATH}/6B.50_words.pkl', 'rb'))
# word2idx = pickle.load(open(f'{GLOVE_PATH}/6B.50_idx.pkl', 'rb'))
# glove = {w: vectors[word2idx[w]] for w in words}
# # weights_matrices = {}
# label = "depression"
# print("Building glove matrix for " + label)
# weights_matrix = build_glove_matrix(glove, voc)
# glove_matrix_pickle = os.path.join("pickles", label + "_" + "glove_matrices.pkl")
# f_matrix = open(glove_matrix_pickle, "wb")
# pickle.dump(weights_matrix, f_matrix, protocol=4)
# f_matrix.close()

# build similar words dictionary
f_vocs = open(VOCS_PICKLE, "rb")
voc = pickle.load(f_vocs)
glove_matrix_pickle = os.path.join("pickles", LABEL + "_" + "glove_matrices.pkl")
f_glove = open(glove_matrix_pickle, "rb")
weights_matrix = pickle.load(f_glove)
sim_words = build_sim_words_dict(voc, weights_matrix)
f_similar_pickle = os.path.join("pickles", LABEL + "_" + "similar_dict.pkl")
f_similar = open(f_similar_pickle, "wb")
pickle.dump(sim_words, f_similar)
f_similar.close()


# # build training data
# f_vocs = open(VOCS_PICKLE, "rb")
# voc = pickle.load(f_vocs)
# label = "depression"
# print("Building training data for " + label)
# glove_matrix_pickle = os.path.join("pickles", label + "_" + "glove_matrices.pkl")
# f_glove = open(glove_matrix_pickle, "rb")
# weights_matrix = pickle.load(f_glove)
# training_data = build_training_data(voc, weights_matrix)
# training_data_pickle = os.path.join("pickles", label + "_" + "training_data")
# f_train = open(training_data_pickle, "wb")
# pickle.dump(training_data, f_train, protocol=4)
