import numpy as np
import torch
from torch import nn, optim
import random
from collections import Counter
import jieba
import matplotlib.pyplot as plt

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

EMBEDING_DIM = 100
EPOCHS = 500
BATCH_SIZE = 5
N_SAMPLES = 3
WINDOWS_SIZE = 5
FREQ = 0
DELETE_WORDS = False
PRINT_EVERY = 100

def preprocess(text, freq):
    result = []
    for t in text:
        words = list(jieba.cut(t))
        word_counts = Counter(words)
        trimmed_words = [word for word in words if word_counts[word] > freq]
        result.append(trimmed_words)
    return result

with open('input/test.txt', 'r', encoding='utf8') as f:
    text = f.read().split('\n')

print("begin data generate")
words = preprocess(text, FREQ)

flatten = lambda l: [item for sublist in l for item in sublist]
vocab = set(flatten(words))
vocab2int = {w: c+1 for c, w in enumerate(vocab)}
vocab2int["[PAD]"] = 0
int2vocab = {c+1: w for c, w in enumerate(vocab)}
int2vocab[0] = "[PAD]"
max_len = max([len(word) for word in words])
for w in words:
    if len(w) < max_len:
        w.extend(["[PAD]"]*(max_len -len(w)))
int_words = [[vocab2int[w] for w in word] for word in words]

int_word_counts = Counter(flatten(int_words))
total_count = len(flatten(int_words))
word_freqs = {w: c/total_count for w, c in int_word_counts.items()}

if DELETE_WORDS:
    t = 1e-5
    prob_drop = {w: 1-np.sqrt(t/word_freqs[w]) for w in int_word_counts}
    train_words = [[w for w in words if random.random() < (1 - prob_drop[w])] for words in int_words]
else:
    train_words = int_words

word_freqs = np.array(list(word_freqs.values()))
unigram_dist = word_freqs / word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (.75)))

def get_target(words, idx, windows_size):
    target_window = np.random.randint(1, windows_size + 1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point: idx] + words[idx + 1: end_point])
    return list(targets)

def get_batch(words, batch_size, windows_size):
    row, col = len(words), len(words[0])
    n = row // batch_size
    for i in range(n + 1):
        batch_x, batch_y = [], []
        batch = words[i * batch_size: (i + 1) * batch_size]
        for i in range(batch_size):
            if i >= len(batch):
                break
            text = batch[i]
            for n in range(len(text)):
                y = get_target(text, n, WINDOWS_SIZE)
                batch_x.extend([text[n]] * len(y))
                batch_y.extend(y)
        yield batch_x, batch_y

class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
    
    def forward_input(self, input_words): 
        input_vectors = self.in_embed(input_words)
        return input_vectors
    
    def forward_output(self, output_words):
        output_vectors = self.in_embed(output_words)
        return output_vectors
    
    def forward_noise(self, size, n_samples):
        noise_dist = self.noise_dist
        noise_words = torch.multinomial(noise_dist, size*n_samples, replacement=True)
        noise_vectors = self.out_embed(noise_words).view(size, n_samples, self.n_embed)
        return noise_vectors


class NegativeSamplelingLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_vectors, output_vectors, noise_vectors):
        size, embed_size = input_vectors.shape
        input_vectors = input_vectors.view(size, embed_size, 1)
        output_vectors = output_vectors.view(size, 1, embed_size)
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)

        return -(out_loss + noise_loss).mean()


model = SkipGramNeg(len(vocab2int), EMBEDING_DIM, noise_dist=noise_dist)
criterion = NegativeSamplelingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print("begin training: ")
steps = 0
for e in range(EPOCHS):
    steps += 1
    for input_words, target_words in get_batch(train_words, BATCH_SIZE, WINDOWS_SIZE):
        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
        input_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        size, _ = input_vectors.shape
        noise_vectors = model.forward_noise(size, N_SAMPLES)
        loss = criterion(input_vectors, output_vectors, noise_vectors)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if steps % PRINT_EVERY == 0:
        print("loss: ", loss)

torch.save(model.state_dict(), "output/model.tc")
for i, w in int2vocab.items():
    vectors = model.state_dict()["in_embed.weight"]
    x, y = float(vectors[i][0]), float(vectors[i][1])
    plt.scatter(x, y)
    plt.annotate(w, xy=(x, y), xytext=(5, 2), textcoords="offset points", ha='right', va="bottom")

plt.savefig('output/words_wight.png')
plt.show()