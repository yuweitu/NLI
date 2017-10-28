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

def loadData(vocab, args):

    keys = list(vocab.keys())
    data = []
    count = 0

    start = time.time()
    print('loading training data... \n')
    with open(args.source+args.file_trn, 'r') as file_list:

        for file in file_list:
            if count < 10:
                line = file.strip().split('\t')
                label = line[0]
                s1 = line[1].strip(' ').split(' ')
                s2 = line[2].split(' ')

                count += 1

                data.append((label, s1, s2))

            if count%10000 == 0:
                print("processed %s docs"%count)
                print("%s seconds elapsed"%(time.time()-start))

    file_list.close()

    print('...training data loaded, total %s samples \n' % len(data))
    return data


class Batch():
    def __init__(self, data, batch_size, vocab):

        assert len(data) > 0
        assert batch_size > 0
        
        self.data = data
        self.vocab = vocab
        self.label_map = {'entailment':0, 'neutral':1, 'contradiction':2}
        self.batch_size = batch_size
        self.start = 0
        
    def __iter__(self):
        return(self)
    
    def __next__(self):

        if self.start >= len(self.data):
            self.start = 0
            raise StopIteration
        
        labels, s1, s2 = self.create_batch(self.data[self.start:self.start+self.batch_size], self.vocab)
        self.start += self.batch_size
        
        return labels, s1, s2

    def create_batch(self, raw_batch, vocab):
        
        #batch inputs
        all_txt = list(zip(*raw_batch))
        print(all_txt)
        idxs = list(map(lambda w: self.label_map[w], all_txt[0]))
        labels = Variable(torch.LongTensor(idxs).view(len(raw_batch),1))
        #print(labels)

        #sentence 1
        idxs = list(map(lambda output: [vocab[w] for w in output], all_txt[1]))
        max_length = np.max(list(map(len, idxs)))
        #print(max_length)
        
        s1 = []
        for idx in idxs:
            temp = np.pad(np.array(idx), 
                          pad_width=((0,max_length-len(idx))), 
                          mode="constant", constant_values=0)
            s1.append(temp)


        s1 = Variable(torch.LongTensor(np.array(s1).transpose()))

        #sentence 2
        idxs = list(map(lambda output: [vocab[w] for w in output], all_txt[2]))
        max_length = np.max(list(map(len, idxs)))
        #print(max_length)
        s2 = []
        for idx in idxs:
            temp = np.pad(np.array(idx), 
                          pad_width=((0,max_length-len(idx))), 
                          mode="constant", constant_values=0)
            s2.append(temp)

        s2 = Variable(torch.LongTensor(np.array(s2).transpose()))

        return labels, s1, s2


class BiLSTM(nn.Module):
    def __init__(self, vocab, emb, args):

        super(BiLSTM, self).__init__()

        self.num_words = len(vocab)
        self.embed_size = emb.shape[1]
        self.hid_size = args.hid
        self.batch_size = args.batch
        self.num_layers = args.num_layers
        ### embedding layer 
        self.emb = nn.Embedding(self.num_words+1, self.embed_size, sparse=True)
        self.emb.weight = Parameter(torch.FloatTensor(emb))      
        
        ### lstm layer
        self.lstm_s1 = nn.LSTM(self.embed_size, self.hid_size, self.num_layers, bias = False, bidirectional=False)

        self.lstm_s2 = nn.LSTM(self.embed_size, self.hid_size, self.num_layers, bias = False, bidirectional=False)

        self.s1_hid = self.init_hidden()
        self.s2_hid = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hid_size)),
                autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hid_size)))

    def forward(self, labels, s1, s2):
       
        use_cuda = self.emb.weight.is_cuda
        #print(s1)
        batch_size = s1.size()[0]

        ### look up the embedding for both sencetences
        s1_emb = self.emb(s1)
        s2_emb = self.emb(s2)
        
        s1_out, self.s1_hid = self.lstm_s1(s1_emb, self.s1_hid)
        s2_out, self.s2_hid = self.lstm_s2(s2_emb, self.s2_hid)

        #print("output")
        #print(s1_out)
        #print("hidden")
        #print(self.s1_hid)

        loss = 0
        return loss.sum() / batch_size


def all_vocab_emb(args):
    emb = np.load(args.source+args.file_emb).item()
    
    vcb_trn = np.load(args.source+args.vocab_trn).item()
    vcb_dev = np.load(args.source+args.vocab_dev).item()
    
    vcb_tst = np.load(args.source+args.vocab_tst).item()
    
    vcb_all = vcb_trn.copy()

    # zero is reserved for padding
    count = len(vcb_trn)+1

    # stack vocab_trn, vocab_dev and vocab_tst
    for i in vcb_dev.keys():
        if i not in vcb_trn.keys():
            vcb_all[i] = count
            count += 1

    for i in vcb_tst.keys():
        if i not in vcb_tst.keys():
            vcb_all[i] = count
            count += 1
    
    vcb_size = len(vcb_all)

    # initialize random embedding
    all_emb = np.random.normal(0, 1, (len(vcb_all), args.emb))
    
    trn_keys = list(vcb_trn.keys())
    count = 0
    
    # replace with pre_trained embedding if exists
    for r in range(len(trn_keys)):
        k = trn_keys[r]
        if type(emb[k]) != int:
            all_emb[r, :] = list(map(float, emb[k]))
            count += 1

    print(all_emb.shape)

    return vcb_all, all_emb 
        

def main(args):
    vocab, emb = all_vocab_emb(args)
    
    batch_size = args.batch
 
    data = loadData(vocab, args)

    random.shuffle(data)
    
    n_batch = int(np.ceil(len(data)/batch_size))

    model = BiLSTM(vocab, emb, args)

    batch = Batch(data, batch_size, vocab)
 
    if args.cuda:
        model.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    losses = []
    
    print("start training model...\n")
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for labels, s1, s2 in batch:
            print(labels)
            print(s1)
            print(s2)

            
            if args.cuda:
                labels, s1, s2 = labels.cuda(), s1.cuda(), s2.cuda()

            if batch.start % 1000 == 0:
                print("training epoch %s: completed %s %%"  % (str(epoch), str(round(100*batch.start/len(data), 2))))

            model.zero_grad()
            loss = model(labels, s1, s2)
            '''
            loss.backward()
            optimizer.step()

            total_loss+=loss.data.cpu().numpy()[0]

        ave_loss = total_loss/n_batch
        print("average loss is: %s" % str(ave_loss))
        losses.append(ave_loss)
        end_time = time.time()
        print("%s seconds elapsed" % str(end_time - start_time))
    '''
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb', type=int, default=300)
    parser.add_argument('-hid', type=int, default=100)
    parser.add_argument('-num_layers', type=int, default=1)
    parser.add_argument('-batch', type=int, default=5)
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-lr', type=float, default =0.05)

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
