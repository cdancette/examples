import os
import torch
import pyphen
from nltk.corpus import cmudict

phen = pyphen.Pyphen(lang='en')

cmu_d = cmudict.dict()


def is_multisyllabic(word):
    if word in cmu_d:
        cmu_syl = [len(list(y for y in x if y[-1].isdigit())) for x in cmu_d[word.lower()]] 
        cmu_test = cmu_syl != [1]
    else:
        cmu_test = False

    return '-' in phen.inserted(word) or cmu_test

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.multisyllabic_ids = [] # mask to delete probabilities

    def add_word(self, word):
        id = len(self.idx2word)
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            if (is_multisyllabic(word) or word in ["'", '"', "<unk>", "=", "<eos>", "@"] or "@" in word):
                self.multisyllabic_ids.append(id)

        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
