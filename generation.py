import os
import torch
import pickle
import re
import unicodedata
from torch.autograd import Variable
import random

USE_CUDA = torch.cuda.is_available()
SRC_FILE = os.path.join("data", "src_file.txt")
MODEL_CHECKPOINT = os.path.join("models", "2019-04-29T19-07-04model.pt")
VOC_PICKLE = os.path.join("pickles", "voc.pkl")
WEIGHTS_MATRIX_PICKLE = os.path.join("pickles", "weights.matrix.pkl")
TEMPERATURE = 1
REPLACE_PROB = 0.3


def load_model():
    model = torch.load(MODEL_CHECKPOINT)
    if USE_CUDA:
        model.cuda()
    else:
        model.cpu()
    return model


def load_voc():
    f_voc = open(VOC_PICKLE, "rb")
    voc = pickle.load(f_voc)
    return voc


def load_weights_matrix():
    f_wm = open(WEIGHTS_MATRIX_PICKLE, "rb")
    weights_matrix = pickle.load(f_wm)
    return weights_matrix


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


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


generated = []
model = load_model()
hidden = model.init_hidden(1)
voc = load_voc()
with open(SRC_FILE, 'rb') as f:
    for line in f:
        words = normalize_string(line.decode("utf-8")).split()
        generated.append(words[0])
        for i in range(1, len(words)):
            prev_word, cur_word = words[i - 1], words[i]
            if (prev_word in voc.word2index) and (cur_word in voc.word2index) and (cur_word in voc.glove_words):
                input = torch.tensor([[voc.word2index[prev_word], voc.word2index[cur_word]]]).cuda()
                output, hidden = model(input, hidden, torch.tensor([1]))
                word_weights = output.squeeze().data.div(TEMPERATURE).exp().cpu()
                word_idx = (torch.multinomial(word_weights, 1)[0]).item()
                # word_idx = output.data.argmax().item()
                if random.uniform(0, 1) < REPLACE_PROB:
                    word = "<" + voc.index2word[word_idx] + ">"
                else:
                    word = cur_word
                generated.append(word)
            else:
                generated.append(cur_word)

        hidden = repackage_hidden(hidden)

print(" ".join(generated))
