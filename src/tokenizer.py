import torch

class Tokenizer():
    def __init__(self):
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.vocab = {}
        self.token_counts = {}
        self.prune_threshold = 5

        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.token_counts[token] = 9999 # set arbitrarily high to prevent pruning

        self.unk_idx = self.vocab['<UNK>']
        self.pad_idx = self.vocab['<PAD>']
        self.bos_idx = self.vocab['<BOS>']
        self.eos_idx = self.vocab['<EOS>']
    
    def tokenize(self, text):
        return text.split()

    def fit_on_text(self, text):
        tokens = self.tokenize(text)
        for token in tokens:
            if token not in self.vocab:
                new_idx = len(self.vocab)
                self.vocab[token] = new_idx
                self.token_counts[token] = 1
            else:
                self.token_counts[token] += 1
    
    # takes in a text stream and returns a 2d tensor of one hot vectors
    def encode(self, text):
        tokens = self.tokenize(text)
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        vocab_size = len(self.vocab)
        hot_vectors = torch.zeros(len(indices), vocab_size)
        for i, idx in enumerate(indices):
            hot_vectors[i, idx] = 1.0
        return hot_vectors

    # removes rare items from vocabulary
    def prune(self):
        for token in self.vocab:
            if self.token_counts[token] < self.prune_threshold:
                self.vocab.pop(token)
    
    def get_vocab_size(self):
        return len(self.vocab)
