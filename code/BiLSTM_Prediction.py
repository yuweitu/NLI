import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from glob import glob
import os
import math
from collections import OrderedDict
import time
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import TSNE
import random
from bilstm import loadData, Batch


def all_vocab_emb(args):

    emb = np.load(args.source + args.file_emb).item()
    vcb_trn = np.load(args.source + args.vocab_trn).item()

    # initialize random embedding
    all_emb = np.random.normal(0, 1, (len(vcb_trn), args.wordemb))

    trn_keys = list(vcb_trn.keys())
    count = 0

    # replace with pre_trained embedding if exists
    for r in range(len(trn_keys)):
        k = trn_keys[r]
        if type(emb[k]) != int:
            all_emb[r, :] = list(map(float, emb[k]))
            count += 1

    print(all_emb.shape)

    return vcb_trn, all_emb


class CharacterRepresentation(nn.Module):
    """Represent each word in a d-dimensional character embedding vector.

    The character-composed embedding is calculated by feeding each character (represented as a character embedding)
    within a word into a Long Short-Term Memory Network ,where the character embeddings are randomly initialized and
    learned jointly with other network parameters from this task.
    We initialize each character as a 20-dimensional vector, and compose each word into a 50-dimensional vector
    with a LSTM layer

    Attributes:
        batch_size: input batch size
        num_words: number of words in each sentence before padding
        word_length: number of characters in the whole character vocabulary
        embed_dim: embedding dimension for each character, default is 20
        char_dim: embedding dimension for each word after RNN layers, default is 50
        n_layers: number of layers in RNN
        dropout: dropout in each layer, default setting to 0.1

    Dimensions:
        Input: batch_size * sentence_length * word_length(len(char_vocab)) * embed_dim(20)
        Output: batch_size * sentence_length * char_dim(50)

    References:
        https://github.com/spro/char-rnn.pytorch/blob/master/model.py


    """

    def __init__(self, batch_size, num_words, word_length, embed_dim=20, char_dim=50, num_layers=1, dropout=0.1):
        super(CharacterRepresentation, self).__init__()
        self.batch_size = batch_size
        self.num_words = num_words
        self.word_length = word_length
        self.embed_dim = embed_dim
        self.char_dim = char_dim
        self.n_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(word_length, embed_dim)

        self.rnn = nn.LSTM(embed_dim, char_dim, num_layers, bias=False, batch_first=True, dropout=dropout)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.rnn(out)
        return out

    def init_hidden(self):
        return (Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size)))


class ContextRepresentation(nn.Module):

    """ Feed a concatated [WordEmbedding, CharEmbedding] into a BiLSTM layer

    Attributes:
        input_size: embed_dim
        hidden_size: default 100
        dropout: default 0.1
    Dimensions:
        Input: batch_size * sequence_length * (word_embedding size + char_dim) = batch_size * sequence_length * (300+50)
        Output: batch_size * sequence_length * (2 * hidden_size)

    """
    def __init__(self, input_size, hidden_size=100,  dropout=0.1):
        super(ContextRepresentation, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bias=False, dropout=dropout, bidirectional = True)

    def forward(self, x):
        out = self.rnn(x)
        return out


class PredictionLayer(nn.Module):

    """ Feed a fixed-length matching vector to probability distribution
    We employ a two layer feed-forward neural network to consume the fixed-length matching vector,
    and apply the softmax function in the output layer.

    Attributes:
        input_size: 4*hidden_size(4*100)
        hidden_size: default 100
        output_size: num of classes, default 3 in our SNLI task
        dropout: default 0.1

    Dimensions:
        Input: batch_size * sequence_length * (4* hidden_size)(4*100)
        output: batch_size * sequence_length * num_classes
    """

    def __init__(self, input_size, hidden_size=100, output_size=3, dropout=0.1):
        super(PredictionLayer, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.dropout(out)
        out = self.softmax(out)
        return out


class BiLSTMPrediction(nn.Module):

    def __init__(self, vocab, emb, args):
        super(BiLSTMPrediction, self).__init__()
        self.batch_size = len(vocab)
        self.num_words = emb.shape[1]
        self.word_length = 26
        # self.word_length = len(char_vocab)
        self.word_emb_dim = args.wordemb
        self.char_emb_dim = args.charemb
        self.embed_dim = self.word_emb_dim + self.char_emb_dim
        self.hidden_size = args.hid
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.num_classes = 3

        # load word embedding
        self.word_emb = nn.Embedding(self.num_words+1, self.word_emb_dim, sparse=True)
        self.word_emb.weight = Parameter(torch.FloatTensor(emb))

        # loading layer
        self.char_representation = CharacterRepresentation(self.batch_size, self.num_words, self.word_length)
        self.context = ContextRepresentation(self.embed_dim, self.hidden_size)
        self.prediction = PredictionLayer(self.hidden_size, self.hidden_size, self.num_classes)

    def forward(self, x1, x2, y1, y2, hidden_size):
        # x1, x2 - words in P, Q; y1, y2 - correpsonding Glove Word Embeddings
        out1 = self.char_representation(x1)
        out2 = self.char_representation(x2)
        out1 = torch.cat([y1, out1], 2)
        out2 = torch.cat([y2, out2], 2)
        #print(out1.size())
        #print(out2.size())

        out1, _ = self.context(out1) # P context
        out2, _ = self.context(out2) # Q context

        out = torch.cat([out1, out2], 2)
        out = self.pre(out)

        return out


def main(args):
    vocab, emb = all_vocab_emb(args)
    batch_size = args.batch
    data = loadData(vocab, args)
    random.shuffle(data)
    n_batch = int(np.ceil(len(data) / batch_size))
    model = BiLSTMPrediction(vocab, emb, args)

    batch = Batch(data, batch_size, vocab)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    losses = []

    print("start training model...\n")
    start_time = time.time()
    for epoch in range(args.epochs):

        model.train()
        total_loss = 0

        for labels, s1, s2 in batch:

            if args.cuda:
                labels, s1, s2 = labels.cuda(), s1.cuda(), s2.cuda()

            if batch.start % 1000 == 0:
                print("training epoch %s: completed %s %%" % (str(epoch), str(round(100 * batch.start / len(data), 2))))

            print(labels)
            print(s1)
            print(s2)
            model.zero_grad()
            out = model(s1, s2)
            loss = loss_func(labels, out)
            loss.backward()
            optimizer.step()

            total_loss+=loss.data.cpu().numpy()[0]

        ave_loss = total_loss/n_batch
        print("average loss is: %s" % str(ave_loss))
        losses.append(ave_loss)
        end_time = time.time()
        print("%s seconds elapsed" % str(end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-wordemb', type=int, default=300)
    parser.add_argument('-charemb', type=int, default=50)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-hid', type=int, default=100)
    parser.add_argument('-num_layers', type=int, default=1)
    parser.add_argument('-batch', type=int, default=5)
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-lr', type=float, default=0.05)

    parser.add_argument('-vocab_trn', type=str)
    parser.add_argument('-vocab_dev', type=str)
    parser.add_argument('-vocab_tst', type=str)
    parser.add_argument('-file_trn', type=str)
    parser.add_argument('-file_dev', type=str)
    parser.add_argument('-file_tst', type=str)
    parser.add_argument('-file_emb', type=str)

    parser.add_argument('-source', type=str)
    parser.add_argument('-saveto', type=str)

    parser.add_argument('-cuda', type=bool, default=False)

    args = parser.parse_args()

    args.vocab_trn = "vocabsnli_trn.npy"
    args.vocab_dev = "vocabsnli_dev.npy"
    args.vocab_tst = "vocabsnli_tst.npy"

    args.file_trn = 'snli_trn.txt'
    args.file_dev = 'snli_dev.txt'
    args.file_tst = 'snli_tst.txt'

    args.file_emb = 'snli_emb.npy'

    args.source = "../intermediate/"
    args.saveto = "../results/"
    args.save_stamp = 'snli_1028'

    args.emb_file = "snli.npy"

    args.cuda = False

    main(args)
