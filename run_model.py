from models.PointerGeneratorModel import PointerGenerator
from models.DGN import DGN
from models.Fact2SeqAttn import Fact2seqAttn
from models.NKLM import NKLM
from TransE import TransE

import torch
import torch.optim as optim
import torch.nn.utils as utils
import torch.cuda as cuda
USE_CUDA = cuda.is_available()

import numpy as np
import argparse
import csv
import json
import os
import re
regex = re.compile('[\.\?\[\](),]+')

from data_utils import Preprocessing
preprocess = Preprocessing()


#*** Training ***

def prepare_input_seq(f):
    f = regex.sub('', f)
    f = f.replace('|', ' ')
    f = f.lower().strip()
    f = f + ' <EOS>'

    words = f.split(' ')
    idx_seq = []
    #print(words)
    for w in words:
        if w in preprocess.word_to_idx:
            idx_seq.append(preprocess.word_to_idx[w])
        else:
            idx_seq.append(preprocess.word_to_idx['<UNK>'])
    return idx_seq


def get_pos_factual_words(facts):
    factual_words = []
    pos_factual_words = []

    for i, f in enumerate(facts):
        f = regex.sub('', f)
        fact_words = []
        pos_idx = []
        if '|' in f:
            rel, obj = f.split('|')
            rel_words = rel.split(' ')
            obj_words = obj.split(' ')
            for w in obj_words:
                if w not in preprocess.stopwords:
                    fact_words.append(w)
                    pos_idx.append(len(rel_words) + obj_words.index(w))
        factual_words.append(fact_words)
        pos_factual_words.append(pos_idx)
    return factual_words, pos_factual_words


def preprocess_description(desc, factual_words, config):
    desc = regex.sub('', desc)
    desc_words = desc.split(' ')
    Y = []
    for w in desc_words:
        is_factual = False
        for fact_idx, fact_words in enumerate(factual_words):
            if w in fact_words:
                Y.append((w, fact_idx))
                is_factual = True
                break
        if not is_factual:
            if w in preprocess.vocabulary:
                Y.append((w, config['max_context_len'] - 1))
            else:
                Y.append(('<UNK>', config['max_context_len'] - 1))
    Y.append(('<EOS>', config['max_context_len'] - 1))
    return Y


def fetch_TransE_embeddings(all_data_file):
    # Fetch the TransE embeddings of (h, r, t) triples
    pos_triples, entity2id, rel2id = preprocess.get_entity_relations(all_data_file)

    transe_model = TransE()
    if USE_CUDA: transe_model = transe_model.cuda()

    transe_model.load_state_dict(torch.load('./TransE.pth')) #
    transe_model.eval()
    print("TransE Model Loaded...")

    ent_embeddings = transe_model.ent_embeddings.weight.data
    rel_embeddings = transe_model.rel_embeddings.weight.data

    ent_transe_embeddings = {}
    for e_idx, e in enumerate(entity2id.keys()):
        ent_transe_embeddings[e] = ent_embeddings[e_idx]

    rel_transe_embeddings = {}
    for r_idx, r in enumerate(rel2id.keys()):
        rel_transe_embeddings[r] = rel_embeddings[r_idx]

    print("TransE embeddings fetched ...")

    return ent_transe_embeddings, rel_transe_embeddings


def get_fact_embeddings(facts, model, config, all_data_file):
    fact_embeddings = torch.zeros(config['max_facts'] - 1, 1, config['fact_embedding_dim'])
    if USE_CUDA: fact_embeddings = fact_embeddings.cuda()
    ent_transe_embeddings, rel_transe_embeddings = fetch_TransE_embeddings(all_data_file)
    for idx, fact in enumerate(facts):
        r, t = fact.split('|')
        if r in rel_transe_embeddings:
            if t in ent_transe_embeddings:
                r_emb = rel_transe_embeddings[r]
                t_emb = ent_transe_embeddings[t]
            else:
                r_emb = rel_transe_embeddings[r]
                t_emb = torch.zeros(100)
            fact_embedding = torch.cat((r_emb, t_emb), dim=0)
            fact_embedding = fact_embedding.unsqueeze(0)
            fact_embeddings[idx] = fact_embedding

    fact_embeddings = torch.cat([fact_embeddings, model.NaF.unsqueeze(0)], dim=0)
    return fact_embeddings


def train(model, optimizer, train_data_file, config, model_name, all_data_file):
    if model_name == 'NKLM': # For NKLM
        with open(train_data_file, 'r') as f:
            batch_fact_embeddings = []
            batch_factual_words = []
            batch_pos_factual_words =[]
            batch_desc = []
            csvreader = csv.reader(f, delimiter=';')
            row_count = 0
            training_loss = 0.0
            for row in csvreader:
                row_count += 1
                facts = row[2: len(row) - 1]
                if len(facts) > config['max_facts']:
                    facts = row[2: config['max_facts']]
                fact_embeddings = get_fact_embeddings(facts, model, config, all_data_file)
                factual_words, pos_factual_words = get_pos_factual_words(facts)

                desc = row[len(row) - 1]
                desc = preprocess_description(desc, factual_words, config)

                batch_fact_embeddings.append(fact_embeddings)
                batch_factual_words.append(factual_words)
                batch_pos_factual_words.append(pos_factual_words)
                batch_desc.append(desc)

                if len(batch_desc) == config['batch_size']:
                    optimizer.zero_grad()
                    batch_loss = model(batch_fact_embeddings, batch_factual_words, batch_pos_factual_words, batch_desc, 'training')
                    training_loss = training_loss + batch_loss.item()
                    batch_loss.backward()
                    utils.clip_grad_value_(model.parameters(), 5)
                    optimizer.step()

                    # Reset for next batch
                    batch_fact_embeddings = []
                    batch_factual_words = []
                    batch_pos_factual_words =[]
                    batch_desc = []

        return training_loss/row_count

    else: # For all other models
        with open(train_data_file,'r') as f:
            csvreader = csv.reader(f, delimiter=';')
            row_count = 0
            training_loss = 0.0
            for row in csvreader:
                #***  CONSTRUCT TENSORS FOR INPUT CONTEXT ***
                facts = row[2:len(row)-1]
                if len(facts) > config['max_context_len']:
                    facts = row[2: config['max_context_len'] + 1]
                factual_words, pos_factual_words = get_pos_factual_words(facts)

                context = np.zeros((config['max_context_len'] - 1, config['max_fact_len']), dtype=np.int)
                for i, fact in enumerate(facts):
                    idx_seq = prepare_input_seq(fact)
                    idx_seq = np.array(idx_seq)
                    context[i] = np.pad(idx_seq, (0, config['max_fact_len'] - len(idx_seq)), 'constant', constant_values = 0)

                context = np.asarray(context, dtype=np.int)
                context = torch.LongTensor(context)
                if USE_CUDA: context = context.cuda()

                # *** CONSTRUCT TENSORS FOR OUTPUT DESCRIPTION ***
                desc = row[len(row)-1]
                if model_name == 'PointerGenerator':
                    description = preprocess_description(desc, factual_words, config)
                elif model_name == 'DGN' or model_name == 'Fact2SeqAttn':
                    desc = prepare_input_seq(desc)
                    desc = np.array(desc)
                    description = torch.LongTensor(desc)
                    if USE_CUDA: description = description.cuda()

                # *** UPDATE THE MODEL WITH THE MINI-BATCH of SIZE 1 ***
                optimizer.zero_grad()
                if model_name == 'PointerGenerator':
                    loss = model(context, factual_words, pos_factual_words, description, 'training')
                elif model_name == 'DGN' or model_name == 'Fact2SeqAttn':
                    loss = model(context, description, 'training')

                loss.backward()
                utils.clip_grad_value_(model.parameters(), 10)
                training_loss += loss.item()
                optimizer.step()

                row_count += 1

        f.close()
        training_loss = training_loss / row_count
        return training_loss


#***  Test ***
def test(model, test_data_file, config, output_file, model_name, all_data_file):
    fout = open(output_file, 'w+')
    if model_name == 'NKLM':
        with open(test_data_file, 'r') as f:
            batch_fact_embeddings = []
            batch_factual_words = []
            batch_pos_factual_words =[]
            batch_desc = []
            csvreader = csv.reader(f, delimiter=';')
            for row in csvreader:
                facts = row[2: len(row) - 1]
                if len(facts) > config['max_facts']:
                    facts = row[2: config['max_facts']]
                fact_embeddings = get_fact_embeddings(facts, model, config, all_data_file)
                factual_words, pos_factual_words = get_pos_factual_words(facts)

                desc = row[len(row) - 1]
                desc = preprocess_description(desc, factual_words)

                batch_fact_embeddings.append(fact_embeddings)
                batch_factual_words.append(factual_words)
                batch_pos_factual_words.append(pos_factual_words)
                batch_desc.append(desc)

                generated_words = model(batch_fact_embeddings, batch_factual_words, batch_pos_factual_words, batch_desc, 'inference')
                generated_desc = ' '.join(generated_words)

                ground_truth = row[len(row) - 1]
                fout.write(ground_truth + ';' +  generated_desc + '\n')

                # Reset for next batch
                batch_fact_embeddings = []
                batch_factual_words = []
                batch_pos_factual_words =[]
                batch_desc = []
        f.close()
    else:
        with open(test_data_file,'r') as f:
            csvreader = csv.reader(f, delimiter=';')
            row_count = 0
            for row in csvreader:
                # *** CONSTRUCT TENSORS FOR INPUT CONTEXT ***
                facts = row[2:len(row)-1]
                if len(facts) > config['max_context_len']:
                    facts = row[2: config['max_context_len'] + 1]
                factual_words, pos_factual_words = get_pos_factual_words(facts)

                context = np.zeros((config['max_context_len'] - 1, config['max_fact_len']), dtype=np.int)
                for i, fact in enumerate(facts):
                    idx_seq = prepare_input_seq(fact)
                    idx_seq = np.array(idx_seq)
                    context[i] = np.pad(idx_seq, (0, config['max_fact_len'] - len(idx_seq)), 'constant', constant_values = 0)

                context = np.asarray(context, dtype=np.int)
                context = torch.LongTensor(context)
                if USE_CUDA: context = context.cuda()

                # *** Prepare Ground Truth ***
                desc = row[len(row)-1]
                description = preprocess_description(desc, factual_words)

                ground_truth = row[len(row) - 1]
                if model_name == 'PointerGenerator':
                    generated_desc = model(context, factual_words, pos_factual_words, description, 'inference')
                elif model_name == 'DGN' or model_name == 'Fact2SeqAttn':
                    generated_desc = model(context, description, 'inference')
                generated_desc = ' '.join(generated_desc)
                fout.write(ground_truth + ';' + generated_desc + '\n')
                row_count += 1
        f.close()
    fout.close()

def main():
    # Instantiate the model
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Input a model name (DGN / Fact2SeqAttn / NKLM / PointerGenerator)')
    parser.add_argument('--datadir', required=True, help="Path to the directory containing the data files")
    parser.add_argument('--config', required=True, help="Input a config file")
    parser.add_argument('--mode', required=True, help="training or eval mode")
    args = parser.parse_args()

    train_data_file = os.path.join(args.datadir, 'training.csv')
    dev_data_file = os.path.join(args.datadir, 'dev.csv')
    test_data_file = os.path.join(args.datadir, 'test.csv')
    all_data_file = os.path.join(args.datadir, 'data10K.csv')

    # Model configurations
    config_file = args.config
    with open(config_file, 'r') as f:
        config = json.load(f)
    print(config)

    model_name = args.model
    print(model_name)

    # Pre-processing
    global preprocess
    if model_name == 'DGN':
        preprocess.get_vocabulary_words(all_data_file)
    if model_name == 'Fact2seqAttn':
        preprocess.get_vocabulary_words(all_data_file)
    if model_name == 'NKLM':
        preprocess.get_hf_vocabulary_words(all_data_file)
        preprocess.get_entity_relations(all_data_file)
    if model_name == 'PointerGenerator':
        preprocess.get_hf_vocabulary_words(all_data_file)
    print(len(preprocess.vocabulary))

    if model_name == 'DGN':
        model = DGN(config, preprocess)
    elif model_name == 'Fact2seqAttn':
        model = Fact2seqAttn(config, preprocess)
    elif model_name == 'NKLM':
        model = NKLM(config, preprocess)
    elif model_name == 'PointerGenerator':
        model = PointerGenerator(config, preprocess)
        print(len(preprocess.vocabulary))
    else:
        print('Please choose an appropriate model (DGN / Fact2SeqAttn / NKLM / PointerGenerator)')
    if USE_CUDA: model = model.cuda()

    if args.mode == 'training':
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr = config['learning_rate'])

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of parameters = ",num_params)

        # Train model
        print("Training model...")
        for epoch in range(config['epochs']):
            loss = train(model, optimizer, train_data_file, config, model_name, all_data_file)
            print("epoch = {} loss = {}".format(epoch + 1, loss))

        # Save model
        if model_name == 'DGN':
            torch.save(model.state_dict(), './DGN.pth')
        elif model_name == 'Fact2seqAttn':
            torch.save(model.state_dict(), './Fact2Seq.pth')
        elif model_name == 'NKLM':
            torch.save(model.state_dict(), './NKLM.pth')
        elif model_name == 'PointerGenerator':
            torch.save(model.state_dict(), './PGM.pth')

    elif args.mode == 'eval':
        # Load saved model
        if model_name == 'DGN':
            model.load_state_dict(torch.load('./DGN.pth'))
        elif model_name == 'Fact2seqAttn':
            model.load_state_dict(torch.load('./Fact2Seq.pth'))
        elif model_name == 'NKLM':
            model.load_state_dict(torch.load('./NKLM.pth'))
        elif model_name == 'PointerGenerator':
            model.load_state_dict(torch.load('./PGM.pth'))

        # Evaluate the model
        print("Evaluating model...")
        if model_name == 'DGN':
            output_file = 'DGN.csv'
        elif model_name == 'Fact2seqAttn':
            output_file = 'Fact2seqAttn.csv'
        elif model_name == 'NKLM':
            output_file = 'NKLM.csv'
        elif model_name == 'PointerGenerator':
            output_file = 'PointerGenerator.csv'

        model.eval()
        test(model, test_data_file, config, output_file, model_name, all_data_file)

if __name__ == '__main__':
    main()
