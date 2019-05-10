import tensorflow as tf

MAX_LEN = 50
SOS_ID = 1


def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)

    dataset = tf.data.Dataset.zip((src_data, trg_data))

    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)

if __name__ == '__main__':
    dataset = MakeDataset('./en-zh/train.tags.en-zh.en')
    print(dataset)