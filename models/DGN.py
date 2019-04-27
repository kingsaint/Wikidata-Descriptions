import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda
import torch.nn.utils as utils

import csv
import numpy as np
from numpy import inf

USE_CUDA = cuda.is_available()

#*** INPUT MODULE ***

class PositionalFactEncoder(nn.Module):
    def __init__(self):
        super(PositionalFactEncoder, self).__init__()

    def forward(self, embedded_sentence):

        _, slen, elen = embedded_sentence.size()

        l = [[(1 - s/(slen-1)) - (e/(elen-1)) * (1 - 2*s/(slen-1)) for e in range(elen)] for s in range(slen)]
        l = torch.FloatTensor(l)
        l = l.unsqueeze(0)
        l = l.expand_as(embedded_sentence)
        l = Variable(l)
        if USE_CUDA: l = l.cuda()
        weighted = embedded_sentence * l
        encoded_output = torch.sum(weighted, dim=1).squeeze(1)
        return encoded_output

class InputModule(nn.Module):
    def __init__(self, config, preprocess):
        super(InputModule, self).__init__()
        self.word_embeddings = nn.Embedding(num_embeddings = len(preprocess.vocabulary), embedding_dim = config['embedding_dim'], padding_idx=0)
        self.fact_encoder = PositionalFactEncoder()

    def forward(self, context):
        context_len, max_fact_len = context.size()
        embedded_context = self.word_embeddings(context)
        embedded_context = embedded_context.view(context_len, max_fact_len, -1)
        encoded_facts = self.fact_encoder(embedded_context)
        return encoded_facts

#*** MEMORY MODULE ***

class MemoryModule(nn.Module):
    def __init__(self, config):
        super(MemoryModule, self).__init__()
        self.linear_1 = nn.Linear(2 * config['hidden_dim'], config['hidden_dim'])
        self.linear_2 = nn.Linear(config['hidden_dim'], 1)
        self.linear_3 = nn.Linear(3 * config['hidden_dim'], config['hidden_dim'])

    def forward(self, m, output_hidden, fact_embeddings):
        # The following two variables will be used later for concatenation
        m_prev = m
        output_context = output_hidden.squeeze(0)

        # resize the tensors
        m = m.expand_as(fact_embeddings)
        output_hidden = output_hidden.squeeze(0).expand_as(fact_embeddings)
        z = torch.cat([torch.abs(fact_embeddings - output_hidden), torch.abs(fact_embeddings - m)], dim=1)

        g = self.linear_2(torch.tanh(self.linear_1(z)))
        g = g.view(-1)
        g = F.softmax(g, dim=0)
        print(g)
        g = g.view(-1, 1)

        # *** 'SOFT ATTENTION' ***
        c = torch.sum(g * fact_embeddings, dim=0).view(1,-1)
        # ************** UPDATE MEMORY ***************************
        m = F.relu(self.linear_3(torch.cat([m_prev, c, output_context], dim=1)))
        return c, m, g

#*** OUTPUT MODULE ***

class OutputModule(nn.Module):
    def __init__(self, config):
        super(OutputModule, self).__init__()
        self.config = config
        self.gru = nn.GRU(input_size = self.config['embedding_dim'] + self.config['hidden_dim'] , hidden_size = self.config['hidden_dim'])

    def forward(self, decoder_input, hidden):
        decoder_input = decoder_input.unsqueeze(1)
        gru_out, hidden = self.gru(decoder_input, hidden)
        return gru_out, hidden

#*** DESCRIPTION GENERATION NETWORK ***

class DGN(nn.Module):
    def __init__(self, config, preprocess):
        super(DGN, self).__init__()
        self.config = config
        self.preprocess = preprocess
        self.word_embeddings = nn.Embedding(num_embeddings = len(self.preprocess.vocabulary), embedding_dim = self.config['embedding_dim'], padding_idx=0)
        self.input_module = InputModule(config, preprocess)
        self.memory_module = MemoryModule(config)
        self.output_module = OutputModule(config)
        self.init_hidden_n_memory = nn.Linear(self.config['max_desc_len'] * self.config['hidden_dim'], self.config['hidden_dim'])
        self.output_1 = nn.Linear(self.config['embedding_dim'] + self.config['hidden_dim'], self.config['hidden_dim'])
        self.output_2 = nn.Linear(self.config['hidden_dim'], len(self.preprocess.vocabulary))
        self.loss_function = nn.NLLLoss()

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, 1, self.config['hidden_dim']))
        if USE_CUDA: h_0 = h_0.cuda()
        return h_0

    def forward(self, context, description, flag):
        fact_embeddings = self.input_module(context)

        # Initialize the hidden state of the output sequence
        output_hidden =  self.init_hidden()
        desc_len = description.size()[0]

        if flag == 'training':
            # Obtain the embedding of the input word
            word_input = Variable(torch.LongTensor([[1]]))
            if USE_CUDA: word_input = word_input.cuda()
            embedding = self.word_embeddings(word_input).squeeze(1)

            # Initialize memory
            m = F.relu(self.init_hidden_n_memory(fact_embeddings.view(1, -1)))
            if USE_CUDA: m = m.cuda()
            # Initialize the training loss
            loss = Variable(torch.FloatTensor([0.0]))
            if USE_CUDA: loss = loss.cuda()
            #******************** Unfold the output sequence *************
            for idx in range(desc_len):

                decoder_input = torch.cat((embedding, m), dim=1 )
                gru_out, output_hidden = self.output_module(decoder_input, output_hidden)
                #***************** Update the memory*********************
                c, m, g = self.memory_module(m, output_hidden, fact_embeddings)
                gru_out = gru_out.view(1,-1)
                output = torch.cat((gru_out, c), dim=1)
                output = torch.tanh(self.output_1(output))
                output = self.output_2(output)
                output = F.log_softmax(output, dim=1)

                #***************** Calculate Loss ***********************
                y_true = description[idx]
                y_true = y_true.view(-1)
                #print(y_true)
                y_pred = output
                #print(y_pred)
                loss += self.loss_function(y_pred, y_true)

                #***************** Prepare Next Decoder Input **************************
                word_input = description[idx].view(1,-1)
                if USE_CUDA: word_input = word_input.cuda()
                embedding = self.word_embeddings(word_input).squeeze(1)
            return loss

        if flag == 'inference':
            word_input = Variable(torch.LongTensor([[1]]))
            if USE_CUDA: word_input = word_input.cuda()
            embedding = self.word_embeddings(word_input).squeeze(1)
            # Initialize memory
            m = F.relu(self.init_hidden_n_memory(fact_embeddings.view(1, -1)))
            if USE_CUDA: m = m.cuda()
            decoded_words = []
            for idx in range(self.config['max_desc_len']):
                decoder_input = torch.cat((embedding, m), dim=1)
                gru_out, output_hidden = self.output_module(decoder_input, output_hidden)

                #*** Update the memory ***
                c, m, g = self.memory_module(m, output_hidden, fact_embeddings)
                gru_out = gru_out.view(1,-1)
                output = torch.cat((gru_out, c), dim=1)
                output = torch.tanh(self.output_1(output))
                output = self.output_2(output)
                output = F.log_softmax(output, dim=1)

                # Interpret the decoder output
                value, index =  output.data.topk(1)
                index = index.data.cpu().numpy()[0][0]
                #*** Prepare Next Decoder Input ***
                word_input = Variable(torch.LongTensor([[index]]))
                if USE_CUDA: word_input = word_input.cuda()
                embedding = self.word_embeddings(word_input).squeeze(1)
                g = g.view(1,-1)
                if index == self.preprocess.word_to_idx['<EOS>']:
                    break
                else:
                    decoded_words.append(self.preprocess.index_to_word[index])
            return decoded_words
