import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layers_num, vocab_size, embedding_layer, batch_size):
        super(LSTMGenerator, self).__init__()
        self.word_embeddings = embedding_layer
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, layers_num)
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.batch_size = batch_size

    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2word.bias.data.fill_(0)
        self.hidden2word.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.word_embeddings(input)
        emb = emb.view(-1, self.batch_size, 600)

        output, hidden = self.lstm(emb, hidden)
        voc_space = self.hidden2word(output.view(output.size(0) * output.size(1), output.size(2)))
        return voc_space.view(output.size(0), output.size(1), voc_space.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.layers_num, bsz, self.hidden_dim).zero_()),
                Variable(weight.new(self.layers_num, bsz, self.hidden_dim).zero_()))
