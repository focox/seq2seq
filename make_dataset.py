import tensorflow as tf


SRC_TRAIN_DATA = './train.en'
TRG_TRAIN_DATA = './train.zh'
CHECKPOINT_PATH = './model/seq2seq_ckpt'

HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 10000
BATCH_SIZE = 100
NUM_EPOCHS = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True


MAX_LEN = 50
SOS_ID = 0

def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda x: tf.string_split([x]).values)
    dataset = dataset.map(lambda x: tf.string_to_number(x, tf.int32))
    dataset = dataset.map(lambda x: (x, tf.size(x)))  # 注意这里有(), 相当于替换成了 (x, size(x))
    return dataset


def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)

    # 处理之后的数据变为：((x, x_len), (y, y_len); (x, x_len), (y, y_len);(...)， 所以每一个样本为：(x, x_len), (y, y_len)
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    def FileterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        # Todo: hardly understood. 长度小于1且大于MAX_LEN的样本直接全部舍弃
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))  # 长度小于1且大于MAX_LEN的舍弃
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))  # 长度小于1且大于MAX_LEN的舍弃
        return tf.logical_and(src_len_ok, trg_len_ok)

    # dataset.filter(predicate), return: Dataset, the Dataset containing the elements of this dataset for which predicate is True
    dataset = dataset.filter(FileterLength)

    # Todo:每一个样本为: (x, x_len), (y, y_len)
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        # Todo: 将标签即目标语言由:'<sos> X Y Z'处理成:'X Y Z <eos>'.
        # Todo: 解码器的输入(trg_input),形式如同'<sos> X Y Z', 解码器的目标输出(trg_label), 形式如同'X Y Z <eos>'
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    dataset = dataset.map(MakeTrgInput)

    # shuffle(buffer_size, sedd=None, reshuffle_each_iteration=None) buffer_size新数据集将从原来数据集中采样的个数。
    dataset = dataset.shuffle(10000)

    # 要与output对应: ((src_input, src_len), (trg_input, trg_label, trg_len)), 会自动补零，补到与同一个batch最长的序列一样的长度。
    padded_shapes = ((tf.TensorShape([None]), tf.TensorShape([])), (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])))
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


if __name__ == '__main__':
    # data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)

    dataset = MakeDataset('./train.en')
    with tf.Session() as tf:
        tf.run(dataset)
