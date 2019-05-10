import codecs
import collections
from operator import itemgetter

RAW_DATA = './simple-examples/data/ptb.train.txt'
VOCAB_OUTPUT = './simple-examples/data/ptb.vocab'

counter = collections.Counter()

with codecs.open(RAW_DATA, mode='r', encoding='utf-8') as f:
    for line in f:
        for word in line.split():
            counter[word] += 1

sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

sorted_words = ['<eos>'] + sorted_words

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as f:
    for word in sorted_words:
        f.write(word + '\n')


def read_data(file_path):
    with open(file_path, 'r') as fin:
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list
