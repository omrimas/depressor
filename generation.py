import os
import torch
import pickle
import re
import unicodedata
from torch.autograd import Variable
import random
import spacy

USE_CUDA = torch.cuda.is_available()
SRC_FILE = os.path.join("data", "src_file.txt")
MODEL_CHECKPOINT = os.path.join("models", "2019-05-05T01-16-53model.pt")
VOC_PICKLE = os.path.join("pickles", "voc.pkl")
WEIGHTS_MATRIX_PICKLE = os.path.join("pickles", "weights.matrix.pkl")
TEMPERATURE = 1
REPLACE_PROB = 0.5
POS_TO_REPLACE = ["ADV", "ADJ", "NOUN", "VERB"]
nlp = spacy.load('en_core_web_md')


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


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'^https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)  # remove links
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
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
        tokens = nlp(normalize_string(line.decode("utf-8")))
        generated.append(tokens[0].text)
        for i in range(1, len(tokens)):
            prev_token, cur_token = tokens[i - 1], tokens[i]

            # check if we can feed it to our network - both words should be in vocabulary
            if (prev_token.text in voc.word2index) and (cur_token.text in voc.word2index):

                # replace flag should be set to
                replace_flag = 0
                # if cur_token.pos_ in POS_TO_REPLACE:
                #     replace_flag = 1

                input = torch.tensor([[voc.word2index[prev_token.text], voc.word2index[cur_token.text]]]).cuda()
                output, hidden = model(input, hidden, torch.tensor([1]), torch.FloatTensor([[replace_flag]]).cuda())
                word_weights = output.squeeze().data.div(TEMPERATURE).exp().cpu()
                word_idx = (torch.multinomial(word_weights, 1)[0]).item()
                # word_idx = output.data.argmax().item()

                output_word = voc.index2word[word_idx]
                rand = random.uniform(0, 1)
                # replace_word = (replace_flag == 1) and (rand < REPLACE_PROB) and (len(output_word) > 1)
                # replace_word = (replace_flag == 1) and (rand < REPLACE_PROB)
                replace_word = (rand < REPLACE_PROB)
                word = "<" + output_word + ">" if replace_word else cur_token.text

                generated.append(word)

            else:
                # previous or current word is not in vocabulary - just add token, and reset hidden
                generated.append(cur_token.text)
                hidden = repackage_hidden(hidden)

        hidden = repackage_hidden(hidden)

print(" ".join(generated))
