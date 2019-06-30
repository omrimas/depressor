EOS_TOKEN = 0  # End-of-sentence token


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {'<PAD>': 0}
        self.word2count = {}
        self.index2word = {0: '<PAD>'}
        self.num_words = 1
        self.glove_words = set()

    def add_sentence(self, sentence):
        indexes = [self.add_word(w) for w in sentence.split()]
        return indexes

    def add_word(self, word, is_glove=False):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1
            self.word2count[word] = 1
            if is_glove:
                self.glove_words.add(word)
        else:
            self.word2count[word] += 1
        return self.word2index[word]

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {'<PAD>': 0}
        self.word2count = {}
        self.index2word = {0: '<PAD>'}
        self.num_words = 1

        for word in keep_words:
            self.add_word(word)
