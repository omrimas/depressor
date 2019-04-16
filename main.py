import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math
import time
from datetime import datetime
import os
import data
import model
import pickle

LABEL = "depression"
USE_CUDA = torch.cuda.is_available()
BATCH_SIZE = 20
device = torch.device("cuda" if USE_CUDA else "cpu")
SEQUENCE_LEN = 10
STARTED_DATE_STRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
MODEL_DIR = "models/" + str(STARTED_DATE_STRING)
VOCS_PICKLE = os.path.join("pickles", "vocs.pkl")

# Model Params
EMBEDDING_SIZE = 50
HIDDEN_SIZE = 128
LAYERS_NUM = 6

# Training Params
GRADIENT_CLIP = 0.5
LOG_INTERVAL = 200
INITIAL_LEARNING_RATE = 20
EPOCHS = 25

# MODEL SAVE DIRECTORY
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print("INITIALIZING Directory: " + MODEL_DIR)

###############################################################################
# Load data
###############################################################################

training_data_pickle = os.path.join("pickles", LABEL + "_" + "training_data")
f_train = open(training_data_pickle, "rb")
training_data, targets = pickle.load(f_train)
training_data = torch.tensor(training_data)

def batchify(data, bsz):
    print("batchify")
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if USE_CUDA:
        data = data.cuda()
    return data


train_data = batchify(training_data, BATCH_SIZE)
print("\nTraining batch size: " + str(BATCH_SIZE))

###############################################################################
# Build the model
###############################################################################

def create_embedding_layer(non_trainable=False):
    glove_matrix_pickle = os.path.join("pickles", LABEL + "_" + "glove_matrices.pkl")
    f_glove = open(glove_matrix_pickle, "rb")
    weights_matrix = pickle.load(f_glove)

    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.tensor(weights_matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer

f_vocs = open(VOCS_PICKLE, "rb")
voc = pickle.load(f_vocs)
voc_size = voc.num_words
print("\n# of tokens in vocabulary: " + str(voc_size))

glove_embedding = create_embedding_layer(True)
model = model.LSTMGenerator(EMBEDDING_SIZE, HIDDEN_SIZE, LAYERS_NUM, voc_size, glove_embedding)
if USE_CUDA:
    model.cuda()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


###############################################################################
# Training code
###############################################################################

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, GRADIENT_CLIP / (totalnorm + 1e-6))


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(SEQUENCE_LEN, len(source) - 1 - i)
    data = Variable(source[i:i + seq_len], volatile=evaluation)
    target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
    return data, target


def train():
    total_loss = 0
    start_time = time.time()
    ntokens = voc_size
    hidden = model.init_hidden(BATCH_SIZE)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, SEQUENCE_LEN)):
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        optimizer.step()
        # clipped_lr = lr * clip_gradient(model, GRADIENT_CLIP)
        # for p in model.parameters():
        #     p.data.add_(-clipped_lr, p.grad.data)

        total_loss += loss.data

        if batch % LOG_INTERVAL == 0 and batch > 0:
            cur_loss = total_loss.item() / LOG_INTERVAL
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // SEQUENCE_LEN, lr,
                              elapsed * 1000 / LOG_INTERVAL, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = INITIAL_LEARNING_RATE
prev_val_loss = None
val_loss = 0.0

if prev_val_loss and val_loss > prev_val_loss:
    if lr > 0.01:
        lr /= 4

print("Learning rate: " + str(lr))

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train()
    # val_loss = evaluate(val_data)

    t1 = '-' * 89
    t2 = '\n| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | valid ppl {:8.2f}\n'.format(epoch, (
            time.time() - epoch_start_time), val_loss, math.exp(val_loss))
    t3 = '-' * 89
    ta = t1 + t2 + t3
    print(ta)

    # SAVE MODEL
    model_name = MODEL_DIR + '/model-{:s}-emsize-{:d}-nhid_{:d}-nlayers_{:d}-batch_size_{:d}-epoch_{:d}'.format(
        "LSTM", EMBEDDING_SIZE, HIDDEN_SIZE, LAYERS_NUM, BATCH_SIZE, epoch) + '.pt'
    print("SAVING: " + model_name)
    print('=' * 89)
    print(" ")
    torch.save(model, model_name)
