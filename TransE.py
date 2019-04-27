import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
import random
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, help="Path to the directory containing the data files")
args = parser.parse_args()
all_data_file = os.path.join(args.datadir, 'data10K.csv')

from data_utils import Preprocessing
p = Preprocessing()
pos_triples, entity2id, rel2id = p.get_entity_relations(all_data_file)

# Hyper-parameters
embedding_size = 100
num_entity = len(entity2id)
num_rel = len(rel2id)
batch_size = 100
learning_rate = 0.001
epochs = 1000

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


class TransE(nn.Module):
    def __init__(self):
        super(TransE, self).__init__()

        ent_weight = FloatTensor(num_entity, embedding_size)
        torch.nn.init.uniform_(ent_weight, a= -6/math.sqrt(embedding_size), b= 6/math.sqrt(embedding_size))
        rel_weight = FloatTensor(num_rel, embedding_size)
        torch.nn.init.uniform_(rel_weight, a= -6/math.sqrt(embedding_size), b= 6/math.sqrt(embedding_size))

        normalized_ent_weight = F.normalize(ent_weight, p=2, dim=1)
        normalized_rel_weight = F.normalize(rel_weight, p=2, dim=1)

        self.ent_embeddings = nn.Embedding(num_entity, embedding_size)
        self.rel_embeddings = nn.Embedding(num_rel, embedding_size)
        self.ent_embeddings.weight = nn.Parameter(normalized_ent_weight)
        self.rel_embeddings.weight = nn.Parameter(normalized_rel_weight)

    def forward(self, batch_pos_h, batch_pos_t, batch_r, batch_neg_h, batch_neg_t):

        batch_pos_h_emb = self.ent_embeddings(batch_pos_h)
        batch_pos_t_emb = self.ent_embeddings(batch_pos_t)
        batch_r_emb = self.rel_embeddings(batch_r)
        batch_neg_h_emb = self.ent_embeddings(batch_neg_h)
        batch_neg_t_emb = self.ent_embeddings(batch_neg_t)

        margin = Variable(FloatTensor([1.0 for j in range(batch_size)]), requires_grad=True)
        if USE_CUDA: margin = margin.cuda()
        zeros = Variable(FloatTensor([0.0 for j in range(batch_size)]), requires_grad=True)
        if USE_CUDA: zeros = zeros.cuda()

        pos_distance = torch.norm(batch_pos_h_emb + batch_r_emb - batch_pos_t_emb, p=2, dim=1)
        neg_distance = torch.norm(batch_neg_h_emb + batch_r_emb - batch_neg_t_emb, p=2, dim=1)

        loss = torch.sum(torch.max(zeros, pos_distance + margin - neg_distance))


        return loss

def prepare_batch(batch_triples):
    #print(batch_triples)
    batch_pos_h = []
    batch_pos_t = []
    batch_r = []
    batch_neg_h = []
    batch_neg_t = []
    for triple in batch_triples:
        h = triple[0]
        r = triple[1]
        t = triple[2]
        #print(h, r, t)
        batch_pos_h.append(entity2id[h])
        batch_pos_t.append(entity2id[t])
        batch_r.append(rel2id[r])

        g = np.random.random()
        if g < 0.5: # replace the t with t'
            while True:
                population = list(set(entity2id.keys()) - set([h, t]))
                neg_t = random.sample(population, 1)[0]
                neg_triple = (h, r, neg_t)
                if neg_triple not in pos_triples.keys():
                    break
            #print(neg_t)
            batch_neg_h.append(entity2id[h])
            batch_neg_t.append(entity2id[neg_t])

        else: # replace h with h'
            while True:
                population = list(set(entity2id.keys()) - set([h, t]))
                neg_h = random.sample(population, 1)[0]
                neg_triple = (neg_h, r, t)
                if neg_triple not in pos_triples.keys():
                    break
            #print(neg_h)
            batch_neg_h.append(entity2id[neg_h])
            batch_neg_t.append(entity2id[t])

    if len(batch_pos_h) < batch_size: # for padding the last batch
        batch_pos_h = batch_pos_h + [0 for i in range(batch_size - len(batch_pos_h))]
        batch_pos_t = batch_pos_t + [0 for i in range(batch_size - len(batch_pos_t))]
        batch_r = batch_r + [0 for i in range(batch_size - len(batch_r))]
        batch_neg_h = batch_neg_h + [0 for i in range(batch_size - len(batch_neg_h))]
        batch_neg_t = batch_neg_t + [0 for i in range(batch_size - len(batch_neg_t))]

    batch_pos_h = LongTensor(batch_pos_h)
    batch_pos_t = LongTensor(batch_pos_t)
    batch_r = LongTensor(batch_r)
    batch_neg_h = LongTensor(batch_neg_h)
    batch_neg_t = LongTensor(batch_neg_t)

    return batch_pos_h, batch_pos_t, batch_r, batch_neg_h, batch_neg_t

def train(model, optimizer):
    batch_triples = []
    training_loss = 0.0
    for triple in pos_triples:
        batch_triples.append(triple)
        if len(batch_triples) == batch_size:
            batch_pos_h, batch_pos_t, batch_r, batch_neg_h, batch_neg_t = prepare_batch(batch_triples)
            batch_loss = model(batch_pos_h, batch_pos_t, batch_r, batch_neg_h, batch_neg_t)
            optimizer.zero_grad()
            batch_loss.backward()

            training_loss += batch_loss.item()
            optimizer.step()
            # Reset batch
            batch_triples = []

    # For the final batch
    batch_pos_h, batch_pos_t, batch_r, batch_neg_h, batch_neg_t = prepare_batch(batch_triples)
    batch_loss = model(batch_pos_h, batch_pos_t, batch_r, batch_neg_h, batch_neg_t)
    optimizer.zero_grad()
    batch_loss.backward()
    training_loss += batch_loss.item()
    for p in model.parameters():
        p.grad.data.clamp_(0, 40)
    optimizer.step()

    model.ent_embeddings.weight.data = F.normalize(model.ent_embeddings.weight.data, p=2, dim=1)

    return training_loss

def main():

    print("Number of entities = ", num_entity)
    print("Number of relations = ", num_rel)
    model = TransE()
    if USE_CUDA: model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("TransE training in progress...")
    for epoch in range(epochs):
        epoch_loss = train(model, optimizer)
        print("epoch={}, loss={}".format(epoch + 1, epoch_loss))
        torch.save(model.state_dict(), './TransE.pth')


if __name__ == '__main__':
    main()


