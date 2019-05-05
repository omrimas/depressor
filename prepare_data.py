import torch
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
import spacy

GLOVE_PATH = os.path.join("data", "glove")
TRAIN_DATA_PATH = os.path.join("data", "training")
VOC_PICKLE = os.path.join("pickles", "voc.pkl")
P = 0.2
MIN_COUNT_WORD = 3
FIXED_TRAINING = os.path.join("data", "fixed_training")
FIXED_TRAINING_TRIMMED = os.path.join("data", "fixed_training_trimmed")
LABEL = "depression"
nlp = spacy.load('en_core_web_md', disable=['parser', 'tagger', 'ner'])


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# def normalize_string(s):
#     s = unicode_to_ascii(s.lower().strip())
#     s = re.sub(r'^https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
#     s = re.sub(r"\<.*?\>", " ", s)
#     # s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-z.,!?']+", r" ", s)
#     # s = re.sub(r"[^a-zA-Z]+", r" ", s)
#     s = re.sub(r"\s+", r" ", s).strip()
#     return s

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'^https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)  # remove links
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def get_post():
    post_i = 0
    with open(TRAIN_DATA_PATH, 'r') as f:
        for line in f:
            line_json = json.loads(line)
            posts, cur_label = line_json[0]["posts"], line_json[0]["label"]

            if cur_label != "depression":
                continue

            for _, post in posts:
                post_i += 1
                if post_i % 50000 == 0:
                    print("Finished %s posts" % post_i)
                yield post


def build_voc(label):
    vocab = voc.Voc(label)

    for post in get_post():
        tokens = nlp(normalize_string(post))
        for token in tokens:
            vocab.add_word(token.text)

    return vocab


def get_fixed_post():
    for post in get_post():
        tokens = nlp(normalize_string(post))
        fixed_post = [token.text for token in tokens if not token.is_space or token.is_bracket]
        yield fixed_post


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


def build_training_data(voc, sim_words):
    input, target = [], []
    prev_word = ""

    with open(FIXED_TRAINING, 'r') as f:
        for idx, line in enumerate(f):
            if (idx % 500 == 0) and (idx > 0):
                print("Finished %s lines" % idx)

            words = line.split()
            for i, w in enumerate(words):
                if prev_word == "":
                    prev_word = w
                    continue

                input.append((voc.word2index[prev_word], sim_words[voc.word2index[w]]))
                target.append(voc.word2index[w])
                prev_word = w

        return input, target


def filter_data(voc):
    lines = []
    with open(FIXED_TRAINING, 'r') as f:
        for idx, line in enumerate(f):
            if idx % 100 == 0:
                print("Finished %s lines" % idx)

            keep_line = True
            for w in line.split():
                if w not in voc.word2index:
                    keep_line = False

            if keep_line:
                lines.append(line)

    return lines


# # build vocs
# print("Building vocabulary...")
# voc = build_voc("depression")
# voc.trim(MIN_COUNT_WORD)
# f = open(VOCS_PICKLE, "wb")
# pickle.dump(voc, f)
# f.close()
# print("Done building vocabulary!")

# print("Building fixed training...")
# str = ""
# with open(FIXED_TRAINING, 'w+') as outf:
#     for i, fixed_post in enumerate(get_fixed_post()):
#         str += " ".join(fixed_post) + "\n"
#         if i != 0 and i % 1000 == 0:
#             outf.write(str)
#             str = ""
# outf.close()
# print("Done building fixed training!")

print("Filter sentences with trimmed words...")
f = open(VOC_PICKLE, "rb")
vocab = pickle.load(f)
lines = filter_data(vocab)
with open(FIXED_TRAINING_TRIMMED, 'w+') as outf:
    for line in lines:
        outf.write(line)
outf.close()

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

# # build similar words dictionary
# f_vocs = open(VOCS_PICKLE, "rb")
# voc = pickle.load(f_vocs)
# glove_matrix_pickle = os.path.join("pickles", LABEL + "_" + "glove_matrices.pkl")
# f_glove = open(glove_matrix_pickle, "rb")
# weights_matrix = pickle.load(f_glove)
# sim_words = build_sim_words_dict(voc, weights_matrix)
# f_similar_pickle = os.path.join("pickles", LABEL + "_" + "similar_dict.pkl")
# f_similar = open(f_similar_pickle, "wb")
# pickle.dump(sim_words, f_similar)
# f_similar.close()


# # # build training data
# f_vocs = open(VOCS_PICKLE, "rb")
# voc = pickle.load(f_vocs)
# f_similar_pickle = os.path.join("pickles", LABEL + "_" + "similar_dict.pkl")
# f_similar = open(f_similar_pickle, "rb")
# sim_words = pickle.load(f_similar)
# print("Building training data for " + LABEL)
# training_data = build_training_data(voc, sim_words)
# training_data_pickle = os.path.join("pickles", LABEL + "_" + "training_data")
# f_train = open(training_data_pickle, "wb")
# pickle.dump(training_data, f_train, protocol=4)
