import torch
import os
import pickle
import model
from torch import nn
import torch.optim as optim
import annoy
import random
from torch.autograd import Variable
from datetime import datetime
import itertools
import spacy

WEIGHTS_MATRIX_PICKLE = os.path.join("pickles", "weights.matrix.pkl")
VOC_PICKLE = os.path.join("pickles", "voc.pkl")
TRAIN_FILE = os.path.join("data", "fixed_training_trimmed")
VEC_LENGTH = 300
ANNOY_INDEX_FILE = os.path.join("data", "glove.6B.300d.txt.annoy")
ANNOY_RESULTS = 5
USE_CUDA = torch.cuda.is_available()
STARTED_DATE_STRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
MODEL_DIR = "models/" + str(STARTED_DATE_STRING)
PAD_TOKEN = 0
POS_TO_REPLACE = ["ADV", "ADJ", "NOUN", "VERB"]

# model params
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 128
LAYERS_NUM = 4
BATCH_SIZE = 25
LEARNING_RATE = 0.0001

# retrain existing model
CONT_TRAIN_MODEL = True
MODEL_CHECKPOINT = os.path.join("models", "2019-05-05T01-16-53model.pt")


def load_model():
    model = torch.load(MODEL_CHECKPOINT)
    if USE_CUDA:
        model.cuda()
    else:
        model.cpu()
    return model


def load_weights_matrix():
    f_wm = open(WEIGHTS_MATRIX_PICKLE, "rb")
    weights_matrix = pickle.load(f_wm)
    return weights_matrix


def load_voc():
    f_voc = open(VOC_PICKLE, "rb")
    voc = pickle.load(f_voc)
    return voc


def create_embedding_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.tensor(weights_matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer


def get_train_sentence():
    legal = 0
    with open(TRAIN_FILE, 'r') as f:
        for idx, line in enumerate(f):
            if idx % 10000 == 0:
                print("Finished %s lines, legal: %s" % (idx, legal))
            if 1 < len(line.split()) < 20:
                legal += 1
                yield line


def get_batch():
    batch = []
    for line in get_train_sentence():
        batch.append(line)
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = []


def index_sentence(voc, sentence):
    return [voc.word2index[w] for w in sentence.split()]


def load_annoy_index():
    annoy_index = annoy.AnnoyIndex(VEC_LENGTH)
    annoy_index.load(ANNOY_INDEX_FILE)
    return annoy_index


def index_sim_words(voc, annoy_index, weights_matrix, sentence, nlp):
    sim_words = []
    replace_flag = []
    tokens = nlp(sentence)
    for w in tokens:
        if w.is_space:
            continue
        if (w.text in voc.glove_words) and (not w.is_stop) and (w.pos_ in POS_TO_REPLACE):
            sim_words.append(
                (annoy_index.get_nns_by_vector(weights_matrix[voc.word2index[w.text]], ANNOY_RESULTS)[
                    random.randint(0, ANNOY_RESULTS - 1)]))
            replace_flag.append(1)
        else:
            sim_words.append(voc.word2index[w.text])
            replace_flag.append(0)

    return sim_words, replace_flag


def zero_padding(l, fillvalue=PAD_TOKEN):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def get_train_input(indexed_sentence, indexed_similar_words):
    train_inputs = []
    targets = []

    indexed_sim, replace_flag = indexed_similar_words[0], indexed_similar_words[1]
    for i in range(1, len(indexed_sentence)):
        train_inputs += [indexed_sentence[i - 1], indexed_sim[i]]
        targets.append(indexed_sentence[i])

    # train_inputs = torch.tensor(train_inputs).cuda() if USE_CUDA else torch.tensor(train_inputs)
    # targets = torch.tensor(targets).cuda() if USE_CUDA else torch.tensor(targets)
    return train_inputs, replace_flag[1:], targets


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


torch.cuda.empty_cache()
device = torch.device("cuda" if USE_CUDA else "cpu")
voc = load_voc()
weights_matrix = load_weights_matrix()
annoy_index = load_annoy_index()
nlp = spacy.load('en_core_web_md')

print("Creating model...")
if CONT_TRAIN_MODEL:
    model = load_model()
else:
    embedding_layer = create_embedding_layer(weights_matrix, True)
    model = model.LSTMGenerator(EMBEDDING_SIZE, HIDDEN_SIZE, LAYERS_NUM, voc.num_words, embedding_layer)
    if USE_CUDA:
        model.cuda()

hidden = model.init_hidden(BATCH_SIZE)
print("Done creating model...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for j, batch in enumerate(get_batch()):
    batch.sort(key=lambda x: len(x.split(" ")), reverse=True)
    indexed_batch = [index_sentence(voc, line) for line in batch]
    indexed_sim_words_batch = [index_sim_words(voc, annoy_index, weights_matrix, line, nlp) for line in
                               batch]
    lengths = torch.tensor([len(indexes) - 1 for indexes in indexed_batch])

    train_inputs, replace_flags, targets = [], [], []
    for i in range(len(indexed_batch)):
        cur_inputs, cur_replace_flags, cur_targets = get_train_input(indexed_batch[i], indexed_sim_words_batch[i])
        train_inputs.append(cur_inputs)
        replace_flags.append(cur_replace_flags)
        targets.append(cur_targets)

    train_inputs = torch.LongTensor(zero_padding(train_inputs)).t().cuda()
    replace_flags = torch.FloatTensor(zero_padding(replace_flags)).view(-1, 1).cuda()
    targets = torch.tensor(zero_padding(targets)).view(-1).cuda()
    hidden = repackage_hidden(hidden)
    model.zero_grad()
    output, hidden = model(train_inputs, hidden, lengths, replace_flags)
    loss = criterion(output.view(-1, voc.num_words), targets)
    loss.backward()
    optimizer.step()
    if j % 10000 == 0 and j > 0:
        print(loss.data.item())
        model_name = MODEL_DIR + str(j) + "_model.pt"
        torch.save(model, model_name)

model_name = MODEL_DIR + "model.pt"
torch.save(model, model_name)
