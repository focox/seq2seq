import codecs
import collections
from operator import itemgetter
import jieba
from tqdm import tqdm


class PreProcess:
    def __init__(self):
        pass

    def read_data(self, filepath):
        with codecs.open(filepath, mode='r') as f:
            data_list = f.readlines()
        return data_list

    def clean_tags(self, data_list):
        data_list = [i[:-1] for i in data_list if '</' not in i and '>']
        return data_list

    def vocab_counter(self, data_list, enable_segment=False, vocab_size=10000, vocab_save_path='./en.vocab'):
        counter = collections.Counter()
        if not enable_segment:
            data_list = [i.split() for i in data_list]
            print('pre-processing english vocab...')
        else:
            data_list = [jieba.cut(i) for i in data_list]
            print('pre-processing chinese vocab...')
        for s in tqdm(data_list):
            for w in s:
                if w:
                    counter[w] += 1
        vocab = sorted(counter.items(), key=itemgetter(1), reverse=True)
        vocab = [x[0] for x in vocab if x[0] != ' ']
        vocab = vocab[:vocab_size]
        vocab = ['<sos>', '<eos>', '<unk>'] + vocab
        with codecs.open(vocab_save_path, 'w', 'utf-8') as f:
            for v in vocab:
                f.write(v + '\n')
        return vocab

    def generate_english_vocab(self, filename_en):
        data_list = self.read_data(filename_en)
        data_list = self.clean_tags(data_list)
        self.en_vocab = self.vocab_counter(data_list)

    def generate_chinese_vocab(self, filename_zh):
        data_list = self.read_data(filename_zh)
        data_list = self.clean_tags(data_list)
        self.zh_vocab = self.vocab_counter(data_list, enable_segment=True, vocab_save_path='./zh.vocab')

    def generate_vocab(self, filename_en, filename_zh):
        self.generate_english_vocab(filename_en)
        self.generate_chinese_vocab(filename_zh)

    def transform2index(self, corpus_path, vocab_path, save_path, enable_segment):
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            vocab = f.read()
        vocab = vocab.split()

        data_list = self.read_data(corpus_path)
        corpus = self.clean_tags(data_list)

        corpus_index = []
        for s in tqdm(corpus):
            if enable_segment:
                s = ['<sos>'] + [i for i in jieba.cut(s)] + ['<eos>']
            else:
                s = ['<sos>'] + s.split() + ['<eos>']
            temp = [str(vocab.index(w)) if w in vocab else str(vocab.index('<unk>')) for w in s]
            corpus_index.append(' '.join(temp))

        with codecs.open(save_path, 'w', 'utf-8') as f:
            for s in corpus_index:
                f.write(s + '\n')


if __name__ == '__main__':
    filename_en = './en-zh/train.tags.en-zh.en'
    filename_zh = './en-zh/train.tags.en-zh.zh'
    preprocess = PreProcess()
    preprocess.transform2index(filename_en, './en.vocab', './en.index', enable_segment=False)

    # preprocess.read_data(filename_en)