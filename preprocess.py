import codecs
import collections
from operator import itemgetter


class PreProcess:
    def __init__(self, filename_en, filename_zh):
        self.filename_en = filename_en
        self.filename_zh = filename_zh

    def read_data(self, filepath):
        with codecs.open(filepath, mode='r') as f:
            data = f.readlines()
        return data

    def clean_tags(self, data_list):
        data_list = [i[:-1] for i in data_list if '</' not in i and '>']
        return data_list

    def vocab_counter(self, data_list, vocab_size, vocab_save_path):
        counter = collections.Counter
        data_list = [i.split() for i in data_list]
        for s in data_list:
            for w in s:
                counter[w] += 1




if __name__ == '__main__':
    filename_en = './en-zh/train.tags.en-zh.en'
    filename_zh = './en-zh/train.tags.en-zh.zh'
    preprocess = PreProcess(filename_en, filename_zh)

    preprocess.read_data(filename_en)