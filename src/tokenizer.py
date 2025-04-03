import torch

class Tokenizer():
    def __init__(self):
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.vocab = {}
        self.reverse_vocab = {}
        self.token_counts = {}
        self.prune_threshold = 5

        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
            self.token_counts[token] = 9999 # set arbitrarily high to prevent pruning

        self.unk_idx = self.vocab['<UNK>']
        self.pad_idx = self.vocab['<PAD>']
        self.bos_idx = self.vocab['<BOS>']
        self.eos_idx = self.vocab['<EOS>']
    
    def tokenize(self, text):
        return text.split()

    def read(self, text):
        tokens = self.tokenize(text)
        for token in tokens:
            if token not in self.vocab:
                new_idx = len(self.vocab)
                self.vocab[token] = new_idx
                self.reverse_vocab[new_idx] = token
                self.token_counts[token] = 1
            else:
                self.token_counts[token] += 1
    
    # takes in a text stream and returns a 2d tensor of one hot vectors
    def encode(self, text):
        tokens = self.tokenize(text)
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        vocab_size = len(self.vocab)
        one_hot_vectors = torch.zeros(len(indices), vocab_size, dtype=torch.long)
        #dense_vectors = torch.zeros(len(indices), 1, dtype=torch.int)
        for i, idx in enumerate(indices):
            one_hot_vectors[i, idx] = 1
            #dense_vectors[i] = idx
        return one_hot_vectors 

    def decode(self, x):
        expected_shape = (len(self.vocab), 1)
        if x.shape != expected_shape:
            raise Exception(f"input tensor is wrong shape. Expected: {expected_shape}. Got: {x.shape}")

        selection = torch.multinomial (x, num_samples=1)
        return self.reverse_vocab[selection]
    
    # removes rare items from vocabulary
    def prune(self):
        to_remove = []
        for token in self.vocab:
            if self.token_counts[token] < self.prune_threshold:
                to_remove.append(token)

        for token in to_remove:
            idx = self.vocab[token]
            self.vocab.pop(token)
            self.reverse_vocab.pop(idx)
    
    def get_vocab_size(self):
        return len(self.vocab)
