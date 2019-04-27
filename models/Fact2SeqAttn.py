import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.cuda as cuda

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
        return torch.sum(weighted, dim=1).squeeze(1)


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

#*** OUTPUT MODULE ***
# Decoder with Attention
class OutputModule(nn.Module):
    def __init__(self, config, preprocess):
        super(OutputModule, self).__init__()
        self.gru = nn.GRU(input_size = 2 * config['embedding_dim'], hidden_size = config['hidden_dim'])
        self.attn = nn.Linear(config['embedding_dim'] + config['hidden_dim'], config['max_context_len'])
        self.attn_2 = nn.Linear(config['max_context_len'], 1)
        self.output_1 = nn.Linear(config['embedding_dim'] + config['hidden_dim'], config['hidden_dim'])
        self.output_2 = nn.Linear(config['hidden_dim'], len(preprocess.vocabulary))

    def forward(self, decoder_input, decoder_hidden, fact_embeddings):
        hidden = decoder_hidden.squeeze(1).expand_as(fact_embeddings)
        attn_energy = F.tanh(self.attn(torch.cat((fact_embeddings, hidden), dim=1)))
        attn_energy = self.attn_2(attn_energy)
        attn_weights = F.softmax(attn_energy.view(1,-1), dim=1)
        context = torch.mm(attn_weights,fact_embeddings)
        gru_input = torch.cat((decoder_input, context), dim=1).unsqueeze(0)
        gru_out, decoder_hidden = self.gru(gru_input, decoder_hidden)
        gru_out = gru_out.view(1,-1)
        output = torch.cat((gru_out, context), dim=1)
        output = F.tanh(self.output_1(output))
        output = self.output_2(output)
        output = F.log_softmax(output, dim=1)
        return output, decoder_hidden



class Fact2seqAttn(nn.Module):  # Description Generation Network
    def __init__(self, config, preprocess):
        super(Fact2seqAttn, self).__init__()
        self.config = config
        self.preprocess = preprocess
        self.word_embeddings = nn.Embedding(num_embeddings = len(self.preprocess.vocabulary), embedding_dim = config['embedding_dim'], padding_idx=0)
        self.input_module = InputModule(config, preprocess)
        self.output_module = OutputModule(config, preprocess)
        self.loss_function = nn.NLLLoss()

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, 1, self.config['hidden_dim']))
        if USE_CUDA: h_0 = h_0.cuda()
        return h_0

    def forward(self, context, description, flag):
        fact_embeddings = self.input_module(context)

        # Initialize the hidden state of the output sequence
        output_hidden = self.init_hidden()

        desc_len = description.size()[0]

        if flag == 'training':
            # Obtain the embedding of the input word
            word_input = Variable(torch.LongTensor([[1]]))
            if USE_CUDA: word_input = word_input.cuda()
            embedding = self.word_embeddings(word_input).squeeze(1)

            loss = Variable(torch.FloatTensor([0.0]))
            if USE_CUDA: loss = loss.cuda()
            #******************** Unfold the output sequence *************
            for idx in range(desc_len):

                decoder_input = embedding
                output, output_hidden = self.output_module(decoder_input, output_hidden, fact_embeddings)

                #***************** Calculate Loss ***********************
                y_true = description[idx]
                y_true = y_true.view(-1)
                y_pred = output
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

            decoded_words = []
            for idx in range(self.config['max_desc_len']):
                decoder_input = embedding
                output, output_hidden = self.output_module(decoder_input, output_hidden, fact_embeddings)

                # Interpret the decoder output
                value, index =  output.data.topk(1)
                index = index.data.cpu().numpy()[0][0]

                #***************** Prepare Next Decoder Input **************************
                word_input = Variable(torch.LongTensor([[index]]))
                if USE_CUDA: word_input = word_input.cuda()
                embedding = self.word_embeddings(word_input).squeeze(1)

                if index == self.preprocess.word_to_idx['<EOS>']:
                    break
                else:
                    decoded_words.append(self.preprocess.index_to_word[index])

            return decoded_words
