import codecs
import collections
from operator import itemgetter
import jieba
from tqdm import tqdm

EN_VOCAB_SIZE = 10000
ZH_VOCAB_SIZE = 20000


class PreProcess:
    def __init__(self, filename_en='./vocab.en', filename_zh='./vocab.zh'):
        try:
            self.read_vocab(filename_en, filename_zh)
        except:
            print('Preserved vocab file NOT found.')
        pass

    def read_data(self, filepath):
        with codecs.open(filepath, mode='r') as f:
            data_list = f.readlines()
        return data_list

    def clean_tags(self, data_list):
        data_list = [i[:-1] for i in data_list if '</' not in i and '>']
        return data_list

    def vocab_counter(self, data_list, enable_segment=False, vocab_size=EN_VOCAB_SIZE, vocab_save_path='./vocab.en'):
        counter = collections.Counter()
        data_list = [jieba.cut(i) for i in data_list]
        print('pre-processing %s vocab...' % ('chinese' if enable_segment else 'english'))
        for s in tqdm(data_list):
            for w in s:
                if w != ' ' and w != '\n':
                    counter[w] += 1
        vocab = sorted(counter.items(), key=itemgetter(1), reverse=True)
        vocab = [x[0] for x in vocab if x[0] != ' ' and x[0] != '\n']
        vocab = vocab[:vocab_size]
        vocab = ['<sos>', '<eos>', '<unk>'] + vocab
        with codecs.open(vocab_save_path, 'w', 'utf-8') as f:
            for v in vocab:
                f.write(v + '\n')
        return vocab

    def generate_english_vocab(self, filename_en):
        data_list = self.read_data(filename_en)
        # data_list = self.clean_tags(data_list)
        self.en_vocab = self.vocab_counter(data_list)

    def generate_chinese_vocab(self, filename_zh):
        data_list = self.read_data(filename_zh)
        # data_list = self.clean_tags(data_list)
        self.zh_vocab = self.vocab_counter(data_list, vocab_size=ZH_VOCAB_SIZE, enable_segment=True, vocab_save_path='./vocab.zh')

    def generate_vocab(self, filename_en, filename_zh):
        self.generate_english_vocab(filename_en)

        self.generate_chinese_vocab(filename_zh)

    def read_vocab(self, filename_en='./vocab.en', filename_zh='./vocab.zh'):
        with codecs.open(filename_en, 'r', 'utf-8') as f:
            en_vocab = f.read()
        with codecs.open(filename_zh, 'r', 'utf-8') as f:
            zh_vocab = f.read()
        self.en_vocab = [i for i in en_vocab.split() if i]
        self.zh_vocab = [i for i in zh_vocab.split() if i]

    def english2id(self, en_str):
        en_list = [self.en_vocab.index(i) if i in self.en_vocab else self.en_vocab.index('<unk>') for i in jieba.cut(en_str) if i != ' ']
        return en_list

    def id2english(self, en_list):
        ls = [self.en_vocab[i] for i in en_list]
        return ' '.join(ls)

    def chinese2id(self, zh_str):
        zh_list = [self.zh_vocab.index(i) if i in self.zh_vocab else self.zh_vocab.index('<unk>') for i in jieba.cut(zh_str) if i != ' ']
        return zh_list

    def id2chinese(self, zh_list):
        ls = [self.zh_vocab[i] for i in zh_list]
        return ' '.join(ls)

    def transform2index(self, corpus_path, vocab_path, save_path, enable_segment):
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            vocab = f.read()
        vocab = vocab.split()
        # vocab = [i for i in vocab if i]

        data_list = self.read_data(corpus_path)
        # corpus = self.clean_tags(data_list)
        corpus = data_list

        corpus_index = []
        print('pre-processing %s corpus...' % ('chinese' if enable_segment else 'english'))
        for s in tqdm(corpus):
            s = [i for i in jieba.cut(s)] + ['<eos>']
            temp = [str(vocab.index(w)) if w in vocab else str(vocab.index('<unk>')) for w in s]
            corpus_index.append(' '.join(temp))

        with codecs.open(save_path, 'w', 'utf-8') as f:
            for s in corpus_index:
                f.write(s + '\n')


if __name__ == '__main__':
    filename_en = './en-zh/train.tags.en-zh.en'
    filename_zh = './en-zh/train.tags.en-zh.zh'
    preprocess = PreProcess()
    preprocess.generate_vocab(filename_en, filename_zh)
    preprocess.transform2index(filename_en, './vocab.en', './train.en', enable_segment=False)
    preprocess.transform2index(filename_zh, './vocab.zh', './train.zh', enable_segment=True)


    # preprocess.read_data(filename_en)