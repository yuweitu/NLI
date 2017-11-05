"""
A basic outline for BiMPM model
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CharacterRepresentation(nn.module):

	"""Represent each word in a d-dimensional character embedding vector.

	The character-composed embedding is calculated by feeding each character (represented as a character embedding) 
	within a word into a Long Short-Term Memory Network ,where the character embeddings are randomly initialized and 
	learned jointly with other network parameters from this task.
	We initialize each character as a 20-dimensional vector, and compose each word into a 50-dimensional vector 
	with a LSTM layer
	
	Attributes:
		sentence_length: number of words in each sentence after padding
		word_length: number of characters in each word after padding
		embed_dim: embedding dimension for each character, default is 20
		char_dim: embedding dimension for each word after RNN layers, default is 50
		ruu unit: LSTM or GRU(default LSTM)
    	n_layers: number of layers in RNN
    	dropout: drouput in each layrer is default setting to 0.1

	Dimensions:
		Input: batch_size * sentence_length * 
		Output: batch_size * sentence_length * char_dim

	References:
		https://github.com/spro/char-rnn.pytorch/blob/master/model.py

	TODOs:
    	* concat character embedding and pre-trained word embedding as the input of BiLSTM layer

    """


	    def __init__(self, senteninput_size, embed_dim, char_dim, rnn_unit="lstm", n_layers=1, dropout = 0.1):
        	
        	super(WordRepresentation, self).__init__()
        	self.sentence_length = sentence_length
        	self.word_length = word_length
        	self.embed_dim= embed_dim
        	self.char_dim = char_dim
        	self.n_layers = n_layers

        	self.embedding = nn.Embedding(word_length, embed_dim)

        	if self.rnn_unit == "gru":
            	self.rnn = nn.GRU(embed_dim, char_dim, rnn_layers,
            		bias = False,
            		batch_first = True,
            		dropout = dropout)

        	elif self.model == "lstm":
            	self.rnn = nn.lstm(embed_dim, char_dim, rnn_layers,
            		bias = False,
            		batch_first = True,
            		dropout = dropout)

    	def forward(self, x):

    		batch_size = input.size(0)
    		out = =self.embedding(x)
    		out, _  = self.rnn(out)
    		return out


    	def init_hidden(self, batch_size):
        	
        	if self.rnn_unit == "lstm":
            	return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        	elif self.rnn_unit == "gru":
        		return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))




class ContextRepresentation(nn.module):

	""" Feed a concatated[CWordEmbedding, harEmbedding] into a BiLSTM layer

	Attributes:
    	input_dim: embed_dim
        hidden_dim: default 100
        rnn_unit: 'lstm' or 'gru'
        dropout: default 0.1


	Dimensions:
    	Input: batch_size * sequence_length * (word_embedding size + rnn_dim)
    	output: batch_size * sequence_length * (2 * hidden_size)
	"""
	def _init_(self, input_size, hidden_size =100, rnn_unit = 'lstm', dropout = 0.1):
		super(ContextRepresentation, self).__init__()
		if rnn_unit == 'lstm':
			self.rnn = nn.lstm(input_size, hidden_size, bias = False, dropout = dropout, bidirectional = True)
		elif rnn_unit == 'gru':
			self.rnn = nn.GRU(input_size, hidden_size, bias = False, dropout = dropout, bidirectional = True)

	def forward(self, x):
		out = self.rnn(x)
		return out


class MultiPerspectiveLayer(nn.module):
    def __nit__(self, )


class PredictionLayer(nn.Module):
    def __init__(self, pre_in, hidden_size, pre_out, dropout = 0.1):
        super(PredictionLayer, self).__init__()
        self.linear1 = nn.Linear(pre_in,hidden_size)
        self.linear2 = nn.Linear(hidden_size,pre_out)
        self.dropout = nn.Dropout(p = dropout)
        self.softmax = nn.Softmax()

    def forward(self,x):
        out = self.dropout(self.linear2(self.linear1(x)))

        return self.softmax(out)



class BiMPM(nn.Module):
    def __init__(self, embedding_dim, sequence_length, nb_chars, nb_per_word, char_embedding_dim, \
                 rnn_dim, rnn_layers, perspective, hidden_size = 100, epsilon = 1e-6, num_classes = 2, rnn_unit = 'gru'):
        super(BiMPM,self).__init__()
        self.hidden_size = hidden_size
        self.rnn_unit = rnn_unit
        self.char_representation = Charrepresentation(sequence_length, nb_chars, \
            nb_per_word, char_embedding_dim, rnn_dim, rnn_layers)
        self.contex_rep = ContextRepresentation(embedding_dim, hidden_size)
        self.multiperspective = MultiPerspective(hidden_size, epsilon, perspective)
        self.aggregation = ContextRepresentation(4 * perspective, hidden_size)
        self.pre = PredictionLayer(4 * hidden_size, hidden_size, num_classes)




    def forward(self, x1, x2 ,y1, y2, hidden_size):
        out1 = self.char_representation(y1)
        out2 = self.char_representation(y2)
        out1 = torch.cat([x1,out1], 2)
        out2 = torch.cat([x2,out2], 2)
        #print(out1.size())
        out1,_ = self.contex_rep(out1)
        out2,_ = self.contex_rep(out2)
        out3 = self.multiperspective(out1, out2)
        out4 = self.multiperspective(out2, out1)
        out3,_ = self.aggregation(out3)
        out4,_ = self.aggregation(out4)
        #timestep x batch x (2*hidden_size)
        pre_list = []
        pre_list.append(out3[:,-1,:hidden_size])
        pre_list.append(out3[:,0,hidden_size:])
        pre_list.append(out4[:,-1,:hidden_size])
        pre_list.append(out4[:,0,:hidden_size])
        pre1 = torch.cat(pre_list,-1)
        # batch x (4*hidden_size)
        out = self.pre(pre1)

        return out



