import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch.cuda as cuda
USE_CUDA = cuda.is_available()

class NKLM(nn.Module):
    def __init__(self, config, p):
        super(NKLM, self).__init__()
        self.config = config
        self.p = p

        NaF_weight = torch.FloatTensor(1, self.config['fact_embedding_dim'])
        if USE_CUDA: NaF_weight = NaF_weight.cuda()
        torch.nn.init.uniform_(NaF_weight, a= -6/math.sqrt(self.config['fact_embedding_dim']), b= 6/math.sqrt(self.config['fact_embedding_dim']))
        self.NaF = nn.Parameter(NaF_weight, requires_grad=True)

        emb_weight = torch.FloatTensor(len(p.vocabulary), self.config['word_embedding_dim'])
        if USE_CUDA: emb_weight = emb_weight.cuda()
        torch.nn.init.uniform_(emb_weight, a= -6/math.sqrt(self.config['word_embedding_dim']), b= 6/math.sqrt(self.config['word_embedding_dim']))
        self.embedding = nn.Embedding(len(p.vocabulary), self.config['word_embedding_dim'])
        self.embedding.weight = nn.Parameter(emb_weight)

        self.decoder_LSTM = nn.LSTM(self.config['fact_embedding_dim'] + self.config['word_embedding_dim'] + self.config['max_fact_len'], self.config['hidden_dim'], self.config['num_layers'], dropout=self.config['p_dropout'])

        self.f_factkey_layer_1 = nn.Linear(self.config['hidden_dim'] + self.config['fact_embedding_dim'], self.config['hidden_dim'])
        self.f_factkey_layer_2 = nn.Linear(self.config['hidden_dim'], self.config['fact_embedding_dim'])

        self.f_copy_layer_1 = nn.Linear(self.config['hidden_dim'] + self.config['fact_embedding_dim'], self.config['hidden_dim'])
        self.f_copy_layer_2 = nn.Linear(self.config['hidden_dim'], 1)

        self.f_voca_layer_1 = nn.Linear(self.config['hidden_dim'] + self.config['fact_embedding_dim'], self.config['hidden_dim'])
        self.f_voca_layer_2 = nn.Linear(self.config['hidden_dim'], self.config['word_embedding_dim'])

        self.f_poskey_layer_1 = nn.Linear(self.config['hidden_dim'] + self.config['fact_embedding_dim'], self.config['hidden_dim'])
        self.f_poskey_layer_2 = nn.Linear(self.config['hidden_dim'], 100)

        self.P = nn.Linear(100, self.config['max_fact_len'])

        self.loss_z = torch.nn.BCELoss()
        self.loss_a = torch.nn.NLLLoss()
        self.loss_w = torch.nn.NLLLoss()

    def forward(self, batch_fact_embeddings, batch_factual_words, batch_pos_factual_words, batch_desc, mode):

        if mode == 'training':
            batch_loss = 0.0
            for b_idx, desc in enumerate(batch_desc):

                # Initial fact input for decoder
                a_t_1 = torch.zeros(1, self.config['fact_embedding_dim'])
                if USE_CUDA: a_t_1 = a_t_1.cuda()

                # Initial hidden state of the decoder
                h_t_1 = torch.zeros(self.config['num_layers'], 1, self.config['hidden_dim'])
                if USE_CUDA: h_t_1 = h_t_1.cuda()

                c_t_1 = torch.zeros(self.config['num_layers'], 1, self.config['hidden_dim'])
                if USE_CUDA: c_t_1 = c_t_1.cuda()

                # Decoder input
                w_v_t_1 = torch.LongTensor([v.vocabulary.index('<SOS>')])
                if USE_CUDA: w_v_t_1 = w_v_t_1.cuda()
                w_v_t_1 = self.embedding(w_v_t_1)

                w_o_t_1 = torch.zeros(1, self.config['max_fact_len'])
                if USE_CUDA: w_o_t_1 = w_o_t_1.cuda()

                # Fact embeddings
                num_facts = len(batch_factual_words[b_idx])
                fact_embeddings = batch_fact_embeddings[b_idx]
                fact_embeddings = fact_embeddings.squeeze(1)

                # Obtain Average fact embeddings
                e_k = torch.mean(fact_embeddings[:num_facts], dim=0).unsqueeze(0)

                for t in range(len(desc)):

                    # INPUT REPRESENTATION
                    x_t = torch.cat((a_t_1, torch.cat((w_v_t_1, w_o_t_1), dim=1)), dim=1)
                    x_t = x_t.unsqueeze(0)

                    # DECODER LSTM
                    o_t, (h_t, c_t) = self.decoder_LSTM(x_t, (h_t_1, c_t_1))

                    # FACT EXTRACTION
                    k_fact = self.f_factkey_layer_1(torch.cat((o_t.squeeze(0), e_k), dim=1))
                    k_fact = F.relu(k_fact)
                    k_fact = self.f_factkey_layer_2(k_fact)

                    prob_a = F.softmax(torch.mm(k_fact, fact_embeddings.t()), dim=1)

                    # Mask the probability
                    mask = [1.0 for i in range(num_facts)] + [0.0 for i in range(self.config['max_facts'] - num_facts - 1)] + [1.0]
                    mask = torch.FloatTensor(mask)
                    mask.requires_grad = True
                    if USE_CUDA: mask = mask.cuda()

                    masked_prob_a = mask * prob_a
                    masked_prob_a = masked_prob_a + self.config['eps']
                    masked_prob_a = masked_prob_a / torch.sum(masked_prob_a)

                    log_masked_prob_a = torch.log(masked_prob_a)

                    # SELECTION OF FACT
                    a_max_idx = torch.argmax(masked_prob_a, dim=1)
                    a_max_idx = a_max_idx.data.cpu().numpy()[0]
                    a_t = fact_embeddings[a_max_idx]
                    a_t = a_t.unsqueeze(0)

                    # SELECTION OF WORD GENERATION SOURCE
                    z_t = self.f_copy_layer_1(torch.cat((o_t.squeeze(0), a_t), dim=1))
                    z_t = F.relu(z_t)
                    z_t = self.f_copy_layer_2(z_t)
                    z_t = torch.sigmoid(z_t)

                    # Ground truth
                    true_z_t = desc[t][2]
                    true_a_t = desc[t][1]
                    true_w_t = desc[t][0]

                    if true_z_t == 0:
                        k_voca = self.f_voca_layer_1(torch.cat((o_t.squeeze(0), a_t), dim=1))
                        k_voca = F.relu(k_voca)
                        k_voca = self.f_voca_layer_2(k_voca)

                        prob_w_t = F.log_softmax(torch.mm(k_voca, self.embedding.weight.t()), dim=1)
                    else:
                        k_pos = self.f_poskey_layer_1(torch.cat((o_t.squeeze(0), a_t), dim=1))
                        k_pos = F.relu(k_pos)
                        k_pos = self.f_poskey_layer_2(k_pos)

                        prob_pos = F.softmax(self.P(k_pos), dim=1)

                        mask = [0.0 for i in range(self.config['max_fact_len'])]
                        for pos_idx in batch_pos_factual_words[b_idx][true_a_t]:
                            mask[pos_idx] = 1.0
                        mask = torch.FloatTensor(mask)
                        mask.requires_grad = True
                        if USE_CUDA: mask = mask.cuda()

                        masked_prob_pos = mask * prob_pos
                        masked_prob_pos = masked_prob_pos + self.config['eps']
                        masked_prob_pos = masked_prob_pos / torch.sum(masked_prob_pos)

                        log_masked_prob_pos = torch.log(masked_prob_pos)

                    # True index of factual / non-factual word
                    if true_z_t == 0:
                        if true_w_t in self.p.vocabulary:
                            true_w_t = self.p.word_to_idx[true_w_t]
                        else:
                            true_w_t = self.p.word_to_idx['<UNK>']
                    else:
                        fact_words = batch_factual_words[b_idx][true_a_t]
                        w_idx = fact_words.index(true_w_t)
                        pos_idx = batch_pos_factual_words[b_idx][true_a_t][w_idx]
                        true_w_t = pos_idx

                    # Tensor of ground truth
                    true_z_t = torch.FloatTensor([true_z_t])
                    if USE_CUDA: true_z_t = true_z_t.cuda()
                    true_a_t = torch.LongTensor([true_a_t])
                    if USE_CUDA: true_a_t = true_a_t.cuda()
                    true_w_t = torch.LongTensor([true_w_t])
                    if USE_CUDA: true_w_t = true_w_t.cuda()

                    if true_z_t == 0:
                        time_step_loss = self.loss_a(log_masked_prob_a, true_a_t) + self.loss_w(prob_w_t, true_w_t) + self.loss_z(z_t, true_z_t)
                    else:
                        time_step_loss = self.loss_a(log_masked_prob_a, true_a_t) + self.loss_w(log_masked_prob_pos, true_w_t) + self.loss_z(z_t, true_z_t)

                    batch_loss += time_step_loss

                    # Prepare input for next time step
                    a_t_1 = a_t
                    h_t_1 = h_t
                    c_t_1 = c_t

                    if true_z_t == 0:
                        w_v_t_1 = true_w_t
                        if USE_CUDA: w_v_t_1 = w_v_t_1.cuda()
                        w_v_t_1 = self.embedding(w_v_t_1)
                        w_o_t_1 = torch.zeros(1, self.config['max_fact_len'])
                        if USE_CUDA: w_o_t_1 = w_o_t_1.cuda()
                    else:
                        w_v_t_1 = torch.zeros(1, self.config['word_embedding_dim'])
                        if USE_CUDA: w_v_t_1 = w_v_t_1.cuda()
                        w_o_t_1 = torch.zeros(1, self.config['max_fact_len'])
                        w_o_t_1[0][pos_idx] = 1.0
                        if USE_CUDA: w_o_t_1 = w_o_t_1.cuda()

            return  batch_loss

        if mode == 'inference':
            for b_idx, desc in enumerate(batch_desc):

                # Initial fact input for decoder
                a_t_1 = torch.zeros(1, self.config['fact_embedding_dim'])
                if USE_CUDA: a_t_1 = a_t_1.cuda()

                # Initial hidden state of the decoder
                h_t_1 = torch.zeros(self.config['num_layers'], 1, self.config['hidden_dim'])
                if USE_CUDA: h_t_1 = h_t_1.cuda()

                c_t_1 = torch.zeros(self.config['num_layers'], 1, self.config['hidden_dim'])
                if USE_CUDA: c_t_1 = c_t_1.cuda()

                # Decoder input
                w_v_t_1 = torch.LongTensor([v.vocabulary.index('<SOS>')])
                if USE_CUDA: w_v_t_1 = w_v_t_1.cuda()
                w_v_t_1 = self.embedding(w_v_t_1)

                w_o_t_1 = torch.zeros(1, self.config['max_fact_len'])
                if USE_CUDA: w_o_t_1 = w_o_t_1.cuda()

                # Fact embeddings
                num_facts = len(batch_factual_words[b_idx])
                fact_embeddings = batch_fact_embeddings[b_idx]
                fact_embeddings = fact_embeddings.squeeze(1)

                # Obtain Average fact embeddings
                e_k = torch.mean(fact_embeddings, dim=0).unsqueeze(0)

                output_seq = []
                for t in range(self.config['max_desc_len']):

                    # INPUT REPRESENTATION
                    x_t = torch.cat((a_t_1, torch.cat((w_v_t_1, w_o_t_1), dim=1)), dim=1)
                    x_t = x_t.unsqueeze(0)

                    # DECODER LSTM
                    o_t, (h_t, c_t) = self.decoder_LSTM(x_t, (h_t_1, c_t_1))

                    # FACT EXTRACTION
                    k_fact = self.f_factkey_layer_1(torch.cat((o_t.squeeze(0), e_k), dim=1))
                    k_fact = torch.nn.functional.relu(k_fact)
                    k_fact = self.f_factkey_layer_2(k_fact)

                    prob_a = F.softmax(torch.mm(k_fact, fact_embeddings.t()), dim=1)

                    mask = [1.0 for i in range(num_facts)] + [0.0 for i in range(self.config['max_facts'] - num_facts - 1)] + [1.0]
                    mask = torch.FloatTensor(mask)
                    mask.requires_grad = True
                    if USE_CUDA: mask = mask.cuda()

                    masked_prob_a = mask * prob_a
                    masked_prob_a = masked_prob_a + self.config['eps']
                    masked_prob_a = masked_prob_a / torch.sum(masked_prob_a)

                    # SELECTION OF FACT
                    a_max_idx = torch.argmax(masked_prob_a, dim=1)
                    a_max_idx = a_max_idx.data.cpu().numpy()[0]
                    a_t = fact_embeddings[a_max_idx]
                    a_t = a_t.unsqueeze(0)

                    # SELECTION OF WORD GENERATION SOURCE
                    z_t = self.f_copy_layer_1(torch.cat((o_t.squeeze(0), a_t), dim=1))
                    z_t = F.relu(z_t)
                    z_t = self.f_copy_layer_2(z_t)
                    z_t = torch.sigmoid(z_t)

                    # WORD GENERATION
                    if z_t < 0.5:
                        k_voca = self.f_voca_layer_1(torch.cat((o_t.squeeze(0), a_t), dim=1))
                        k_voca = F.relu(k_voca)
                        k_voca = self.f_voca_layer_2(k_voca)

                        prob_w = F.log_softmax(torch.mm(k_voca, self.embedding.weight.t()), dim=1)
                        w_max_idx = torch.argmax(prob_w, dim=1)
                        w_max_idx = w_max_idx.data.cpu().numpy()[0]

                        output_word = self.p.index_to_word[w_max_idx]
                        if output_word == '<EOS>':
                            break
                        output_seq.append(output_word)

                    else:

                        k_pos = self.f_poskey_layer_1(torch.cat((o_t.squeeze(0), a_t), dim=1))
                        k_pos = F.relu(k_pos)
                        k_pos = self.f_poskey_layer_2(k_pos)

                        prob_pos = F.softmax(self.P(k_pos), dim=1)

                        mask = [0.0 for i in range(self.config['max_fact_len'])]
                        if a_max_idx < num_facts:
                            for pos_idx in batch_pos_factual_words[b_idx][a_max_idx]:
                                mask[pos_idx] = 1.0
                        mask = torch.FloatTensor(mask)
                        mask.requires_grad = True
                        if USE_CUDA: mask = mask.cuda()

                        masked_prob_pos = mask * prob_pos
                        masked_prob_pos = masked_prob_pos + self.config['eps']
                        masked_prob_pos = masked_prob_pos / torch.sum(masked_prob_pos)

                        pos_max_idx = torch.argmax(masked_prob_pos, dim=1)
                        pos_max_idx = pos_max_idx.data.cpu().numpy()[0]

                        if a_max_idx < num_facts and pos_max_idx in batch_pos_factual_words[b_idx][a_max_idx]:
                            rel_pos_idx = batch_pos_factual_words[b_idx][a_max_idx].index(pos_max_idx)
                            output_word = batch_factual_words[b_idx][a_max_idx][rel_pos_idx]
                        else:
                            output_word = '<UNK>'
                        output_seq.append(output_word)

                    # Prepare input for next time step

                    a_t_1 = a_t
                    h_t_1 = h_t
                    c_t_1 = c_t

                    if z_t < 0.5:
                        w_v_t_1 = torch.LongTensor([w_max_idx])
                        if USE_CUDA: w_v_t_1 = w_v_t_1.cuda()
                        w_v_t_1 = self.embedding(w_v_t_1)

                        w_o_t_1 = torch.zeros(1, self.config['max_fact_len'])
                        if USE_CUDA: w_o_t_1 = w_o_t_1.cuda()

                    else:
                        if a_max_idx < num_facts:
                            w_v_t_1 = torch.zeros(1, self.config['word_embedding_dim'])
                            if USE_CUDA: w_v_t_1 = w_v_t_1.cuda()

                            w_o_t_1 = torch.zeros(1, self.config['max_fact_len'])
                            w_o_t_1[0][pos_max_idx] = 1.0
                            if USE_CUDA: w_o_t_1 = w_o_t_1.cuda()
                        else:
                            w_v_t_1 = torch.LongTensor([self.p.word_to_idx['<UNK>']])
                            if USE_CUDA: w_v_t_1 = w_v_t_1.cuda()
                            w_v_t_1 = self.embedding(w_v_t_1)

                            w_o_t_1 = torch.zeros(1, self.config['max_fact_len'])
                            if USE_CUDA: w_o_t_1 = w_o_t_1.cuda()

            return output_seq
