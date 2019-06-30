import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layers_num, vocab_size, embedding_layer):
        super(LSTMGenerator, self).__init__()
        self.word_embeddings = embedding_layer
        self.lstm = nn.LSTM(embedding_dim * 2 + 1, hidden_dim, layers_num, batch_first=True)
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

        self.hidden_dim = hidden_dim
        self.layers_num = layers_num

    def init_weights(self):
        initrange = 0.1
        # self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2word.bias.data.fill_(0)
        self.hidden2word.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, input_lengths, replace_flags):
        emb1 = self.word_embeddings(input)
        emb2 = emb1.view(-1, 600)
        emb3 = torch.cat((emb2, replace_flags), dim=1)
        emb4 = emb3.view(input.size(0), -1, 601)
        # emb3 = torch.cat((emb2, replace_flags), dim=1)
        # emb4 = emb3.view(-1, input.size(0), 601)
        packed = nn.utils.rnn.pack_padded_sequence(emb4, input_lengths, batch_first=True)

        output, hidden = self.lstm(packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        voc_space = self.hidden2word(output.contiguous().view(output.size(0) * output.size(1), output.size(2)))
        return voc_space.view(output.size(0), output.size(1), voc_space.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.layers_num, bsz, self.hidden_dim).zero_()),
                Variable(weight.new(self.layers_num, bsz, self.hidden_dim).zero_()))
