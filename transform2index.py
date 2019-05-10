import codecs
import sys

VOCAB_PATH = './simple-examples/data/ptb.vocab'
RAW_TRAIN_DATA = './simple-examples/data/ptb.train.txt'
TRAIN_DATA = './simple-examples/data/train_data_index.txt'

RAW_TRAIN_DATA = './simple-examples/data/ptb.test.txt'
TRAIN_DATA = './simple-examples/data/test_data_index.txt'


RAW_TRAIN_DATA = './simple-examples/data/ptb.valid.txt'
TRAIN_DATA = './simple-examples/data/valid_data_index.txt'


with codecs.open(VOCAB_PATH, 'r', 'utf-8') as f:
    vocab = f.readlines()

vocab = [i[:-1] for i in vocab if i]
vocab = {c: idx for idx, c in enumerate(vocab)}

def get_id(word):
    return vocab[word] if word in vocab else vocab['<unk>']


train_data = []

train_index_output = codecs.open(TRAIN_DATA, 'w', 'utf-8')
with codecs.open(RAW_TRAIN_DATA, 'r', 'utf-8') as f:
    for line in f:
        if line:
            sentence = line.split() + ['<eos>']
            out_line = ' '.join([str(get_id(i)) for i in sentence]) + '\n'
            print(out_line)
            train_index_output.write(out_line)
train_index_output.close()


# print(train_data)