from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.cuda as cuda

# Flag to use GPU
USE_CUDA = cuda.is_available()

#*** ENCODER MODULE ***
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

class EncoderModule(nn.Module):
    def __init__(self, config, p):
        super(EncoderModule, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(num_embeddings = len(p.vocabulary),embedding_dim = self.config['embedding_dim'], padding_idx=0)
        self.fact_encoder = PositionalFactEncoder()

    def forward(self, context):
        context_len, max_fact_len = context.size()
        word_embedded_context = self.word_embeddings(context)
        word_embedded_context = word_embedded_context.view(context_len, max_fact_len, -1)
        encoded_facts = self.fact_encoder(word_embedded_context)
        return encoded_facts

#*** DECODER MODULE ***

class DecoderModule(nn.Module):
    def __init__(self, config, p):
        super(DecoderModule, self).__init__()
        self.config = config
        self.p = p
        self.gru = nn.GRU(input_size = 2 * self.config['embedding_dim'] + self.config['max_fact_len'], hidden_size = self.config['hidden_dim'])
        self.attn_1 = nn.Linear(self.config['embedding_dim'] + self.config['hidden_dim'], self.config['max_context_len'])
        torch.nn.init.xavier_normal_(self.attn_1.weight)
        self.attn_2 = nn.Linear(self.config['max_context_len'], 1)
        torch.nn.init.xavier_normal_(self.attn_2.weight)
        self.attn_3 = nn.Linear(self.config['hidden_dim'], self.config['max_fact_len'])
        torch.nn.init.xavier_normal_(self.attn_3.weight)
        self.output_1 = nn.Linear(self.config['embedding_dim'] + self.config['hidden_dim'], self.config['hidden_dim'])
        torch.nn.init.xavier_normal_(self.output_1.weight)
        self.output_2 = nn.Linear(self.config['hidden_dim'], len(p.vocabulary))
        torch.nn.init.xavier_normal_(self.output_2.weight)
        self.output_3 = nn.Linear(self.config['embedding_dim'] + self.config['hidden_dim'], self.config['hidden_dim'])
        torch.nn.init.xavier_normal_(self.output_3.weight)
        self.linear_1 = nn.Linear(self.config['hidden_dim']+ self.config['embedding_dim'], 50)
        torch.nn.init.xavier_normal_(self.linear_1.weight)
        self.linear_2 = nn.Linear(50, 1)
        torch.nn.init.xavier_normal_(self.linear_2.weight)

    def forward(self, decoder_input, decoder_hidden, fact_embeddings, factual_words, pos_factual_words, f_t, flag):
        num_facts = len(pos_factual_words)
        # *** Calculate attention weights ***
        hidden = decoder_hidden
        hidden = hidden.squeeze(1).expand_as(fact_embeddings)

        attn_1_input = torch.cat((fact_embeddings, hidden), dim=1)
        attn_energy = torch.tanh(self.attn_1(attn_1_input))
        attn_energy = self.attn_2(attn_energy).view(1, -1)
        attn_weights = F.softmax(attn_energy, dim=1)

        mask = [1.0 for j in range(num_facts)] + [0.0 for j in range(num_facts, self.config['max_context_len'] - 1)] + [1.0]
        mask = Variable(torch.FloatTensor(mask), requires_grad=True)
        if USE_CUDA: mask = mask.cuda()

        attn_weights = mask * attn_weights
        attn_weights = attn_weights + self.config['eps']
        sum_attn_weights = torch.sum(attn_weights)
        attn_weights = attn_weights / sum_attn_weights
        log_attn_weights = torch.log(attn_weights)

        attn_weighted_facts = torch.mm(attn_weights, fact_embeddings)

        # *** Find the fact with maximum attention ***
        f_max_idx = torch.argmax(attn_weights, dim=1).data.cpu().numpy()[0]
        emb_f_max = fact_embeddings[f_max_idx]
        emb_f_max = emb_f_max.view(1, -1)

        # *** GRU Decoder ***
        gru_input = torch.cat((decoder_input, emb_f_max), dim=1).unsqueeze(0)
        gru_out, decoder_hidden = self.gru(gru_input, decoder_hidden)
        gru_out = gru_out.view(1,-1)

        if flag == 'training':
            if f_t == self.config['max_context_len'] - 1:
                # *** Non-factual word ***
                output = torch.cat((gru_out, attn_weighted_facts), dim=1)
                output = torch.relu(self.output_1(output))
                output = self.output_2(output)
                y_pred = F.log_softmax(output, dim=1)

            else:
                # *** Factual word ***
                if f_t < num_facts and len(pos_factual_words[f_t]) > 0:
                    emb_f_t = fact_embeddings[f_t].view(1, -1)
                    output = torch.cat((gru_out, emb_f_t), dim=1)
                    output = torch.relu(self.output_3(output))

                    word_attn_energy = self.attn_3(output)
                    word_attn_weights = F.softmax(word_attn_energy, dim=1)

                    mask = [0.0 for i in range(self.config['max_fact_len'])]
                    for pos_idx in pos_factual_words[f_t]:
                        mask[pos_idx] = 1.0
                    mask = Variable(torch.FloatTensor(mask), requires_grad = True)
                    if USE_CUDA: mask = mask.cuda()

                    word_attn_weights = mask * word_attn_weights
                    word_attn_weights = word_attn_weights + self.config['eps']
                    sum_word_attn_weights = torch.sum(word_attn_weights)
                    word_attn_weights = word_attn_weights / sum_word_attn_weights

                    y_pred = torch.log(word_attn_weights)
                else:
                    # *** Non-factual word ***
                    output = torch.cat((gru_out, attn_weighted_facts), dim=1)
                    output = torch.relu(self.output_1(output))
                    output = self.output_2(output)
                    y_pred = F.log_softmax(output, dim=1)

            return y_pred, log_attn_weights, decoder_hidden

        if flag == 'inference':
            if f_max_idx == self.config['max_context_len'] - 1:
                # *** Non-factual word ***
                output = torch.cat((gru_out, attn_weighted_facts), dim=1)
                output = torch.relu(self.output_1(output))
                output = self.output_2(output)
                y_pred = F.log_softmax(output, dim=1)

                w_max_idx = torch.argmax(y_pred, dim=1)
                w_max_idx = w_max_idx.data.cpu().numpy()[0]

                output_word = self.p.index_to_word[w_max_idx]
            else:
                # *** Factual word ***
                if f_max_idx < num_facts and len(pos_factual_words[f_max_idx]) > 0:
                    output = torch.cat((gru_out, emb_f_max), dim=1)
                    output = torch.relu(self.output_3(output))

                    word_attn_energy = self.attn_3(output)
                    word_attn_weights = F.softmax(word_attn_energy, dim=1)

                    mask = [0.0 for i in range(self.config['max_fact_len'])]
                    for pos_idx in pos_factual_words[f_max_idx]:
                        mask[pos_idx] = 1.0
                    mask = Variable(torch.FloatTensor(mask), requires_grad = True)
                    if USE_CUDA: mask = mask.cuda()

                    word_attn_weights = mask * word_attn_weights
                    word_attn_weights = word_attn_weights + self.config['eps']
                    sum_word_attn_weights = torch.sum(word_attn_weights)
                    word_attn_weights = word_attn_weights / sum_word_attn_weights

                    pos_max_idx = torch.argmax(word_attn_weights, dim=1)
                    pos_max_idx = pos_max_idx.data.cpu().numpy()[0]

                    if pos_max_idx in pos_factual_words[f_max_idx]:
                        rel_pos_idx = pos_factual_words[f_max_idx].index(pos_max_idx)
                        output_word = factual_words[f_max_idx][rel_pos_idx]
                    else:
                        output_word = '<UNK>'
                else:
                    # *** Non-factual word ***
                    output = torch.cat((gru_out, attn_weighted_facts), dim=1)
                    output = torch.relu(self.output_1(output))
                    output = self.output_2(output)
                    y_pred = F.log_softmax(output, dim=1)

                    w_max_idx = torch.argmax(y_pred, dim=1)
                    w_max_idx = w_max_idx.data.cpu().numpy()[0]

                    output_word = self.p.index_to_word[w_max_idx]

            return output_word, f_max_idx, decoder_hidden

class PointerGenerator(nn.Module):
    def __init__(self, config, p):
        super(PointerGenerator, self).__init__()
        self.config = config
        self.p = p
        self.input_module = EncoderModule(config, p)
        self.output_module = DecoderModule(config, p)
        self.loss_function = nn.NLLLoss()

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, 1, self.config['hidden_dim']))
        if USE_CUDA: h_0 = h_0.cuda()
        return h_0

    def forward(self, context, factual_words, pos_factual_words, description, flag):
        num_facts = len(pos_factual_words)
        fact_embeddings = self.input_module(context)
        mean_fact_embeddings = torch.mean(fact_embeddings[:num_facts], dim=0)
        mean_fact_embeddings = mean_fact_embeddings.view(1, -1)
        fact_embeddings = torch.cat((fact_embeddings, mean_fact_embeddings), dim=0)

        if flag == 'training':
            # Initialize the hidden state of the output sequence h_0
            decoder_hidden = self.init_hidden()

            # Obtain the embedding of the first decoder input word <SOS>
            decoder_word_input = Variable(torch.LongTensor([1]))
            if USE_CUDA: decoder_word_input = decoder_word_input.cuda()
            decoder_word_input = self.input_module.word_embeddings(decoder_word_input)

            decoder_pos_input = Variable(torch.zeros(1, self.config['max_fact_len']))
            if USE_CUDA: decoder_pos_input = decoder_pos_input.cuda()

            desc_len = len(description)

            loss = Variable(torch.FloatTensor([0.0]))
            if USE_CUDA: loss = loss.cuda()

            #*** Rollout the output sequence ***
            for t in range(desc_len):
                decoder_input = torch.cat((decoder_word_input, decoder_pos_input), dim=1)

                f_t = description[t][1]
                y_pred, f_pred, decoder_hidden = self.output_module(decoder_input, decoder_hidden, fact_embeddings, factual_words, pos_factual_words, f_t, 'training')

                # True factual / non-factual word
                y_true = description[t][0]
                if f_t == self.config['max_context_len'] - 1:
                    y_true = self.p.word_to_idx[y_true]
                else:
                    w_idx = factual_words[f_t].index(y_true)
                    pos_idx = pos_factual_words[f_t][w_idx]
                    y_true = pos_idx

                # Calculate Loss
                f_true = Variable(torch.LongTensor([f_t]))
                if USE_CUDA: f_true = f_true.cuda()

                y_true = Variable(torch.LongTensor([y_true]))
                if USE_CUDA: y_true = y_true.cuda()

                curr_loss = self.loss_function(y_pred, y_true) + self.loss_function(f_pred, f_true)
                loss += curr_loss

                # Prepare Next Decoder Input
                if f_t == self.config['max_context_len'] - 1:
                    decoder_word_input = self.input_module.word_embeddings(y_true)
                    decoder_pos_input = Variable(torch.zeros(1, self.config['max_fact_len']))
                    if USE_CUDA: decoder_pos_input = decoder_pos_input.cuda()
                else:
                    decoder_word_input = Variable(torch.zeros(1, self.config['embedding_dim']))
                    if USE_CUDA: decoder_word_input = decoder_word_input.cuda()
                    decoder_pos_input = Variable(torch.zeros(1, self.config['max_fact_len']))
                    if USE_CUDA: decoder_pos_input = decoder_pos_input.cuda()
                    decoder_pos_input[0][pos_idx] = 1.0

            return loss

        if flag == 'inference':
            # Initialize the hidden state of the output sequence
            decoder_hidden = self.init_hidden()

            # Initialize the first input to the decoder <SOS>
            decoder_word_input = Variable(torch.LongTensor([1]))
            if USE_CUDA: decoder_word_input = decoder_word_input.cuda()
            decoder_word_input = self.input_module.word_embeddings(decoder_word_input)

            decoder_pos_input = Variable(torch.zeros(1, self.config['max_fact_len']))
            if USE_CUDA: decoder_pos_input = decoder_pos_input.cuda()

            f_max_idx = self.config['max_context_len'] - 1

            decoded_words = []
            for t in range(self.config['max_desc_len']):
                decoder_input = torch.cat((decoder_word_input, decoder_pos_input), dim=1)
                output_word, f_max_idx, decoder_hidden = self.output_module(decoder_input, decoder_hidden, fact_embeddings, factual_words, pos_factual_words, f_max_idx, 'inference')

                if output_word == '<EOS>':
                    break
                else:
                    if output_word != '<UNK>':
                        decoded_words.append(output_word)

                if f_max_idx == (self.config['max_context_len'] - 1) or output_word == '<UNK>':
                    y_prev = self.p.word_to_idx[output_word]
                    y_prev = Variable(torch.LongTensor([y_prev]))
                    if USE_CUDA: y_prev = y_prev.cuda()
                    decoder_word_input = self.input_module.word_embeddings(y_prev)

                    decoder_pos_input = Variable(torch.zeros(1, self.config['max_fact_len']))
                    if USE_CUDA: decoder_pos_input = decoder_pos_input.cuda()
                else:
                    decoder_word_input = Variable(torch.zeros(1, self.config['embedding_dim']))
                    if USE_CUDA: decoder_word_input = decoder_word_input.cuda()

                    decoder_pos_input = Variable(torch.zeros(1, self.config['max_fact_len']))
                    if USE_CUDA: decoder_pos_input = decoder_pos_input.cuda()

                    if f_max_idx < num_facts and output_word in factual_words[f_max_idx]:
                        w_idx = factual_words[f_max_idx].index(output_word)
                        pos_idx = pos_factual_words[f_max_idx][w_idx]
                        decoder_pos_input[0][pos_idx] = 1.0

            return decoded_words
