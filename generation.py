import os
import torch
import pickle
import re
import unicodedata
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
SRC_FILE = os.path.join("data", "src_file.txt")
MODEL_CHECKPOINT = "models/2019-03-26T12-20-25/model-LSTM-emsize-50-nhid_128-nlayers_6-batch_size_20-epoch_25.pt"
VOC_PICKLE = os.path.join("pickles", "voc.pkl")
WEIGHTS_MATRIX_PICKLE = os.path.join("pickles", "weights.matrix.pkl")


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


model = load_model()
voc = load_voc()
with open(SRC_FILE, 'r', encoding="utf8") as f:
    for line in f:
        words = normalize_string(line).split()
        for i in range(1, len(words)):
            prev_word, cur_word = words[i - 1], words[i]
            if voc.word2index[prev_word] and voc.word2index[cur_word]:
                input = torch.tensor([voc.word2index[prev_word], voc.word2index[cur_word]])
                hidden = repackage_hidden(hidden)
                output, hidden = model(input, hidden)
                print("omri")
