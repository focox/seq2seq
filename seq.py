import tensorflow as tf


SOURCE_VACAB = 10000
TARGET_VACAB = 40000

NUM_LAYER = 2
BATCH_SIZE = 100

HIDDIN_SIZE = 1024
SHARE_EMB_AND_SOFTMAX = True
MAX_CLIP_NORM = 5


class Seq2Seq:
    def __init__(self):
        self.encode_cell = tf.nn.rnn_cell.MultiRNNCell(tf.nn.rnn_cell.BasicLSTMCell(HIDDIN_SIZE) for _ in NUM_LAYER)
        self.decode_cell = tf.nn.rnn_cell.MultiRNNCell(tf.nn.rnn_cell.BasicLSTMCell(HIDDIN_SIZE) for _ in NUM_LAYER)

        with tf.variable_scope('encode_embedding'):
            self.encode_embedding = tf.get_variable('encode_embedding', [SOURCE_VACAB, HIDDIN_SIZE])
        with tf.variable_scope('decode_embedding'):
            self.decode_embedding = tf.get_variable('decode_embedding', [TARGET_VACAB, HIDDIN_SIZE])
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.decode_embedding)
        else:
            self.softmax_weight = tf.get_variable('softmax_weight', [HIDDIN_SIZE, TARGET_VACAB])
        self.softmax_bias = tf.get_variable('softmax_bias', [TARGET_VACAB])

    def forward_backward(self, src_inputs, src_length, target_inputs, target_length, labels):
        """

        :param args: inputs: 4-D tensor with shape [source_batch_size, source_input_size, target_batch_size, target_input_size].
        :return:
        """
        src_inputs = tf.nn.embedding_lookup(self.encode_embedding, src_inputs)
        target_inputs = tf.nn.embedding_lookup(self.decode_embedding, target_inputs)


        with tf.variable_scope('encode'):
            # output, state = tf.nn.dynamic_rnn(cell, input_data, ...), where state is the final state, but output include all every output
            # output is [batch_size, max_time, cell.output_size] 注意这里的 max_time
            self.encode_output, self.encode_state = tf.nn.dynamic_rnn(self.encode_cell, src_inputs, src_length, tf.float32)
        with tf.variable_scope('decode'):
            self.decode_output, self.decode_state = tf.nn.dynamic_rnn(self.decode_cell, target_inputs, target_length, initail_state=self.encode_state)

        output = tf.reshape(self.decode_output, [-1, HIDDIN_SIZE])
        self.output = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        label_weight = tf.sequence_mask(target_length)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels)
        loss = label_weight * loss
        loss_mean = tf.reduce_sum(loss)/tf.reduce_sum(label_weight)

        trainable_variables = tf.trainable_variables()
        grad = tf.gradients(loss_mean, trainable_variables)
        # 此函数还没弄清楚
        grad, _ = tf.clip_by_global_norm(grad, MAX_CLIP_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(zip(grad, trainable_variables))
        return loss_mean, self.train_op




