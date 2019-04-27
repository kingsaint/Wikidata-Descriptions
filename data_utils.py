import csv
import re
regex = re.compile('[\.\?\[\](),]+')


class Preprocessing():
    def __init__(self):
        self.vocabulary = []
        self.word_to_idx = {}
        self.index_to_word = {}
        self.word_count = {}
        self.stopwords = {"i":1, "me":2, "my":3, "myself":4, "we":5, "our":6, "ours":7, "ourselves":8, "you":9, "your":10, "yours":11, "yourself":12,\
             "yourselves":13, "he":14, "him":15, "his":16, "himself":17, "she":18, "her":19, "hers":20, "herself":21, "it":22, "its":23, "itself":24,\
             "they":25, "them":26, "their":27, "theirs":28, "themselves":29, "what":30, "which":31, "who":32, "whom":33, "this":34, "that":35,\
             "these":36, "those":37, "am":38, "is":39, "are":40, "was":41, "were":42, "be":43, "been":44, "being":45, "have":46, "has":47, "had":48, "having":49,\
             "do":50, "does":51, "did":52, "doing":53, "a":54, "an":55, "the":56, "and":57, "but":58, "if":59, "or":60, "because":61, "as":62, "until":63,\
             "while":64, "of":65, "at":66, "by":67, "for":68, "with":69, "about":70, "against":71, "between":72, "into":73, "through":74, "during":75,\
             "before":76, "after":77, "above":78, "below":79, "to":80, "from":81, "up":82, "down":83, "in":84, "out":85, "on":86, "off":87, "over":88,\
             "under":89, "again":90, "further":91, "then":92, "once":93, "here":94, "there":95, "when":96, "where":97, "why":98, "how":99, "all":100,\
             "any":101, "both":102, "each":103, "few":104, "more":105, "most":106, "other":107, "some":108, "such":109, "no":110, "nor":111, "not":112, "only":113,\
             "own":114, "same":115, "so":116, "than":117, "too":118, "very":119, "s":120, "t":121, "can":122, "will":123, "just":124, "don":125, "should":126, "now":127}


    def get_vocabulary_words(self, input_file):
        with open(input_file, 'r') as fin:
            csvreader = csv.reader(fin, delimiter=';')
            for row in csvreader:
                facts = row[2 : len(row) - 1]
                desc = row[len(row) - 1]

                for f in facts:
                    f = regex.sub('', f)
                    rel, obj = f.split('|')
                    obj_words = obj.split(' ')
                    rel_words = rel.split(' ')

                    for w in obj_words:
                        if w not in self.word_count:
                            self.word_count[w] = 1
                        else:
                            self.word_count[w] += 1
                    for w in rel_words:
                        if w not in self.word_count:
                            self.word_count[w] = 1
                        else:
                            self.word_count[w] += 1

                desc = regex.sub('', desc)
                desc_words = desc.split(' ')
                for w in desc_words:
                    if w not in self.word_count:
                        self.word_count[w] = 1
                    else:
                        self.word_count[w] += 1
            fin.close()

            self.vocabulary = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + list(self.word_count.keys())

            for i, w in enumerate(self.vocabulary):
                self.word_to_idx[w] = i
                self.index_to_word[i] = w


    def get_hf_vocabulary_words(self, input_file):
        with open(input_file, 'r') as fin:
            csvreader = csv.reader(fin, delimiter=';')
            for row in csvreader:
                facts = row[2 : len(row) - 1]
                desc = row[len(row) - 1]

                for f in facts:
                    f = regex.sub('', f)
                    rel, obj = f.split('|')
                    obj_words = obj.split(' ')
                    rel_words = rel.split(' ')

                    for w in obj_words:
                        if w not in self.word_count:
                            self.word_count[w] = 1
                        else:
                            self.word_count[w] += 1
                    for w in rel_words:
                        if w not in self.word_count:
                            self.word_count[w] = 1
                        else:
                            self.word_count[w] += 1

                desc = regex.sub('', desc)
                desc_words = desc.split(' ')
                for w in desc_words:
                    if w not in self.word_count:
                        self.word_count[w] = 1
                    else:
                        self.word_count[w] += 1
        fin.close()

        sorted_word_count = sorted(self.word_count.items(), key=lambda kv: kv[1])

        words = []
        for i in range(len(sorted_word_count)):
            words.append(sorted_word_count[i][0])

        self.vocabulary = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + words[::-1][0:1000]

        for i, w in enumerate(self.vocabulary):
            self.word_to_idx[w] = i
            self.index_to_word[i] = w


    def get_entity_relations(self, input_file):
        pos_triples = {}
        entity2id={'PAD': 0}
        rel2id = {'PAD': 0}
        with open(input_file, 'r') as f:
            csvreader = csv.reader(f, delimiter=';')
            for row in csvreader:
                facts = row[2:len(row) - 1]
                sub = row[0]
                for fact in facts:
                    rel, obj = fact.split('|')
                    t = (sub, rel, obj)
                    if t not in pos_triples:
                        pos_triples[t] = 1
                    if sub not in entity2id:
                        entity2id[sub] = len(entity2id)
                    if rel not in rel2id:
                        rel2id[rel] = len(rel2id)
                    if obj not in entity2id:
                        entity2id[obj] = len(entity2id)
        return pos_triples, entity2id, rel2id
