import os
import pickle

import torch
import pandas as pd
import numpy as np
import spacy as sp
from torch import nn
from torch import optim
from torch.autograd import Variable
import random
import torch.nn.functional as F
from sklearn import preprocessing
import sklearn.metrics.pairwise as pairwise
from datetime import datetime, date, time
from collections import Counter, defaultdict
import json
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
import scipy
import warnings

SKIP_LINES = 3880000

warnings.filterwarnings('ignore')

raw = pd.read_csv('data/EmoBank/corpus/raw.tsv', sep='\t')
wr_emos = pd.read_csv('data/EmoBank/corpus/writer.tsv', sep='\t')

joined = raw.set_index('id').join(wr_emos.set_index('id'))


def normalize(data, target_column_name):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data.values.reshape(-1, 1))
    normalized = scaler.transform(data.values.reshape(-1, 1))
    joined[target_column_name] = normalized
    return scaler


arousal_normalizer = normalize(joined["Arousal"], "Arousal_norm")
dominance_normalizer = normalize(joined["Dominance"], "Dominance_norm")
valence_normalizer = normalize(joined["Valence"], "Valence_norm")
joined.head(100)


def get_specific_emotion_values(valence, arousal, dominance):
    anger = np.array([-0.51, 0.59, 0.25])
    disgust = np.array([-0.60, 0.35, 0.11])
    fear = np.array([-0.64, 0.60, -0.43])
    sadness = np.array([-0.63, -0.27, -0.33])
    joy = np.array([0.76, 0.48, 0.35])
    surprise = np.array([0.40, 0.67, -0.13])

    arousal_norm = arousal_normalizer.transform(np.array([arousal]).reshape(-1, 1))[0][0]
    valence_norm = valence_normalizer.transform(np.array([valence]).reshape(-1, 1))[0][0]
    dominance_norm = dominance_normalizer.transform(np.array([dominance]).reshape(-1, 1))[0][0]
    input_arr = np.array([valence_norm, arousal_norm, dominance_norm])

    anger_dist = (input_arr @ anger) / (np.linalg.norm(input_arr) * np.linalg.norm(anger))
    disgust_dist = (input_arr @ disgust) / (np.linalg.norm(input_arr) * np.linalg.norm(disgust))
    fear_dist = (input_arr @ fear) / (np.linalg.norm(input_arr) * np.linalg.norm(fear))
    sadness_dist = (input_arr @ sadness) / (np.linalg.norm(input_arr) * np.linalg.norm(sadness))
    joy_dist = (input_arr @ joy) / (np.linalg.norm(input_arr) * np.linalg.norm(joy))
    surprise_dist = (input_arr @ surprise) / (np.linalg.norm(input_arr) * np.linalg.norm(surprise))
    return anger_dist, disgust_dist, fear_dist, sadness_dist, joy_dist, surprise_dist


get_specific_emotion_values(3.7, 4.2, 3.0)

nlp = sp.load('en_core_web_md')

joined = joined.dropna()


class Vocab:
    def __init__(self, embeddings_file):
        self.vec = []
        self.word_count = 0
        self.ind2word = {}
        self.word2ind = {}
        for line in open(embeddings_file, 'r', encoding="utf8"):
            values = line.split(" ")
            v = []
            for i in range(1, len(values)):
                v.append(float(values[i]))
            self.vec.append(v)
            self.ind2word[self.word_count] = values[0]
            self.word2ind[values[0]] = self.word_count
            self.word_count += 1


lang = Vocab("data/glove.6B.300d.txt")

print(lang.word2ind['table'])

vocab = lang

MAX_LENGTH = 100
MAX_SD = 0.5


def unseen_text2tensor(text, vocab):
    proc_sentence = nlp(text.lower())
    if len(proc_sentence) > MAX_LENGTH:
        return None
    indexes = []
    for t in proc_sentence:
        if t.text in vocab.word2ind:
            indexes.append(vocab.word2ind[t.text])
    if len(indexes) < 3:
        return None
    input_tensor = torch.LongTensor(indexes).cuda()
    return input_tensor


def row2tensor(row, vocab):
    proc_text = nlp(row['sentence'].lower())
    if len(proc_text) > MAX_LENGTH:
        return None
    indexes = []
    for t in proc_text:
        if t.text in vocab.word2ind:
            indexes.append(vocab.word2ind[t.text])
    if len(indexes) == 0:
        return None
    input_tensor = torch.LongTensor(indexes).cuda()
    target_tensor = torch.FloatTensor([float(row['Valence']), float(row['Arousal']), float(row['Dominance'])]).cuda()
    return input_tensor, target_tensor


def data2tensors(data, vocab):
    instances = []
    for index, row in data.iterrows():
        if row['sd.Valence'] > MAX_SD or row['sd.Arousal'] > MAX_SD or row['sd.Dominance'] > MAX_SD:
            continue
        t = row2tensor(row, vocab)
        instances.append(t)
    return instances


data = data2tensors(joined, vocab)

random.choice(data)


class Attn(nn.Module):
    def __init__(self, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.lin = nn.Linear(self.hidden_size * 2, hidden_size * 2)
        self.weight_vec = nn.Parameter(torch.FloatTensor(1, hidden_size * 2))

    def forward(self, outputs):
        seq_len = len(outputs)

        attn_energies = torch.zeros(seq_len).cuda()  # B x 1 x S

        for i in range(seq_len):
            attn_energies[i] = self.score(outputs[i])

        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, output):
        energy = self.lin(output)
        energy = torch.dot(self.weight_vec.view(-1), energy.view(-1))
        return energy


class EmoModel(nn.Module):
    def __init__(self, vocab, hidden_size, output_size, n_layers):
        super(EmoModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding = nn.Embedding(vocab.word_count, len(vocab.vec[0]))
        self.embedding.weight = nn.Parameter(torch.FloatTensor(vocab.vec))
        # self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(len(vocab.vec[0]), hidden_size, n_layers, bidirectional=True)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.attn = Attn(hidden_size)

    def forward(self, input_text):
        seq_len = len(input_text)
        embedded_words = self.embedding(input_text).view(seq_len, 1, -1)
        last_hidden = self.init_hidden()
        rnn_outputs, hidden = self.lstm(embedded_words, last_hidden)
        attn_weights = self.attn(rnn_outputs.squeeze(0))
        attn_weights = attn_weights.squeeze(1).view(seq_len, 1)
        rnn_outputs = rnn_outputs.squeeze(1)
        attn_weights = attn_weights.expand(seq_len, self.hidden_size * 2)
        weigthed_outputs = torch.mul(rnn_outputs, attn_weights)
        output = torch.sum(weigthed_outputs, -2)
        output = self.out(output)
        return output

    def init_hidden(self):
        return (torch.zeros(self.n_layers * 2, 1, self.hidden_size).cuda(),
                torch.zeros(self.n_layers * 2, 1, self.hidden_size).cuda())


n_layers = 3
hidden_size = 1000
MODEL_CHECKPOINT = os.path.join("models", "emotions.filtered.55Kiter.glove.6B.300d.3L.1000hidden.attention.model")

model = EmoModel(vocab, hidden_size, 3, n_layers).cuda()

model.load_state_dict(torch.load(MODEL_CHECKPOINT))
# criterion = nn.MSELoss()
# learning_rate = 0.0001
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

model(data[0][0])


def train_sentence(sentence, target, criterion, model, optimizer):
    optimizer.zero_grad()
    seq_len = len(sentence)
    loss = 0
    output = model(sentence)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return output, loss.item()


def predict_random():
    pair = random.choice(data)
    inp = pair[0]
    tgt = pair[1]
    output = model(inp)
    return output, tgt


def predict_unseen_text(text, model):
    inp = unseen_text2tensor(text, vocab)
    if inp is not None:
        output = model(inp)
    else:
        output = -1
    return output


predict_unseen_text("What a bad day:(", model)

predict_unseen_text("This was a very very bad day I am so depressed", model)

row = predict_unseen_text("This was a very very bad day I am so depressed", model)
print(row)
emotions = get_specific_emotion_values(row[0], row[1], row[2])
print(emotions)

FIXED_TRAINING_TRIMMED = os.path.join("data", "fixed_training_trimmed")
FIXED_TRAINING_TRIMMED_FILTERED = os.path.join("data", "fixed_training_trimmed_filtered")


def flush_all(lines):
    for i, emo in enumerate(['anger', 'disgust', 'fear', 'sadness', 'joy', 'surprise']):
        with open(FIXED_TRAINING_TRIMMED_FILTERED + "_" + emo, 'a') as outf:
            for line in lines[i]:
                outf.write(line)
        outf.close()


def emotions_filter(input_filepath):
    lines = [[], [], [], [], [], []]
    emotions = [0] * 6
    fp_in = open(input_filepath, "r")
    for i, line in enumerate(fp_in):
        if i < SKIP_LINES:
            continue
        if i % 10000 == 0:
            print("done with %s lines" % i)
            print(emotions)

        if i % 50000 == 0 and i != 0:
            print("flushing...")
            flush_all(lines)
            lines = [[], [], [], [], [], []]
            print("done!")

        r = predict_unseen_text(line, model)
        if isinstance(r, int) and (r == -1):
            continue
        emo_vals = get_specific_emotion_values(r[0], r[1], r[2])
        idx = np.asarray(emo_vals).argmax().item()
        lines[idx].append(line)
        emotions[idx] += 1
    fp_in.close()


# emotions_filter(FIXED_TRAINING_TRIMMED)


def add_valence(input_filepath, output_filepath):
    lines = []
    fp_in = open(input_filepath, "r")
    fp_out = open(output_filepath, "w+")
    for i, line in enumerate(fp_in):
        if i % 10000 == 0:
            for line in lines:
                fp_out.write(line)
            lines = []
            print("done with %s lines" % i)

        r = predict_unseen_text(line, model)
        if isinstance(r, int) and (r == -1):
            continue

        lines.append(str(r[0].data.item()) + "#" + line)

    fp_in.close()
    fp_out.close()


add_valence(FIXED_TRAINING_TRIMMED_FILTERED + "_sadness", FIXED_TRAINING_TRIMMED_FILTERED + "_sadness_with_valence")
