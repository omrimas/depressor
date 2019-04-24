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

WEIGHTS_MATRIX_PICKLE = os.path.join("pickles", "weights.matrix.pkl")
VOC_PICKLE = os.path.join("pickles", "voc.pkl")
TRAIN_FILE = os.path.join("data", "fixed_training")
VEC_LENGTH = 300
ANNOY_INDEX_FILE = os.path.join("data", "glove.6B.300d.txt.annoy")
ANNOY_RESULTS = 10
USE_CUDA = torch.cuda.is_available()
STARTED_DATE_STRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
MODEL_DIR = "models/" + str(STARTED_DATE_STRING)
PAD_TOKEN = 0

# model params
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 128
LAYERS_NUM = 4
BATCH_SIZE = 20
LEARNING_RATE = 0.0001


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
            if idx % 500 == 0:
                print("Finished %s lines, legal: %s" % (idx, legal))
            if 1 < len(line.split()) < 30:
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


def index_close_words(voc, annoy_index, weights_matrix, sentence):
    return [(annoy_index.get_nns_by_vector(weights_matrix[voc.word2index[w]], ANNOY_RESULTS)[random.randint(0, 9)]) for
            w in sentence.split()]


def zero_padding(l, fillvalue=PAD_TOKEN):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def get_train_input(indexed_sentence, indexed_close_words):
    train_inputs = []
    targets = []
    for i in range(1, len(indexed_sentence)):
        train_inputs += [indexed_sentence[i - 1], indexed_close_words[i]]
        targets.append(indexed_sentence[i])

    # train_inputs = torch.tensor(train_inputs).cuda() if USE_CUDA else torch.tensor(train_inputs)
    # targets = torch.tensor(targets).cuda() if USE_CUDA else torch.tensor(targets)
    return train_inputs, targets


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


device = torch.device("cuda" if USE_CUDA else "cpu")
voc = load_voc()
weights_matrix = load_weights_matrix()
annoy_index = load_annoy_index()

print("Creating model...")
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
    indexed_sim_words_batch = [index_close_words(voc, annoy_index, weights_matrix, line) for line in batch]
    lengths = torch.tensor([len(indexes)-1 for indexes in indexed_batch])

    train_inputs, targets = [], []
    for i in range(len(indexed_batch)):
        cur_inputs, cur_targets = get_train_input(indexed_batch[i], indexed_sim_words_batch[i])
        train_inputs.append(cur_inputs)
        targets.append(cur_targets)

    train_inputs = torch.LongTensor(zero_padding(train_inputs)).t().cuda()
    targets = torch.tensor(zero_padding(targets)).view(-1).cuda()
    hidden = repackage_hidden(hidden)
    model.zero_grad()
    output, hidden = model(train_inputs, hidden, lengths)
    loss = criterion(output.view(-1, voc.num_words), targets)
    loss.backward()
    optimizer.step()
    if j % 1000 == 0 and j > 0:
        print(loss.data.item())
        model_name = MODEL_DIR + str(j) + "_model.pt"
        torch.save(model, model_name)

# for i, sentence in enumerate(get_train_sentence()):
#     indexed_sentence = index_sentence(voc, sentence)
#     indexed_close_words = index_close_words(voc, annoy_index, weights_matrix, sentence)
#     train_inputs, targets = get_train_input(indexed_sentence, indexed_close_words)
#     hidden = repackage_hidden(hidden)
#     model.zero_grad()
#     output, hidden = model(train_inputs, hidden)
#     loss = criterion(output.view(-1, voc.num_words), targets)
#     loss.backward()
#     optimizer.step()
#     if i % 10000 == 0 and i > 0:
#         print(loss.data.item())
#         model_name = MODEL_DIR + str(i) + "_model.pt"
#         torch.save(model, model_name)
#
model_name = MODEL_DIR + "model.pt"
torch.save(model, model_name)
