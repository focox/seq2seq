import tensorflow as tf
from make_dataset import *
from preprocess import *

SRC_TRAIN_DATA = './train.en'
TRG_TRAIN_DATA = './train.zh'
CHECKPOINT_PATH = './model/seq2seq_ckpt'

HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 20000
BATCH_SIZE = 100
NUM_EPOCHS = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True

SOS_ID = 0
EOS_ID = 1


class NMTModel:
    def __init__(self):
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

        self.src_embedding = tf.get_variable('src_embedding', [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable('trg_embedding', [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable('weight', [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable('softmax_bias', [TRG_VOCAB_SIZE])

    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        # tf.nn.dynamic_rnn(
        #     cell,
        #     inputs,
        #     sequence_length=None,
        #     initial_state=None,
        #     dtype=None,
        #     parallel_iterations=None,
        #     swap_memory=False,
        #     time_major=False,
        #     scope=None
        # )
        # Returns: A pair(outputs, state)
        # Todo: tf.nn.dynamic_rnn()可以动态处理补了零之后的序列，如何处理？
        with tf.variable_scope('encoder'):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, src_size, dtype=tf.float32)

        with tf.variable_scope('decoder'):
            dec_outputs, _ = tf.nn.dynamic_rnn(self.dec_cell, trg_emb, trg_size, initial_state=enc_state)

        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]), logits=logits)

        label_weights = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        # Todo: 有啥用
        cost_per_token = cost / tf.reduce_sum(label_weights)

        trainable_variables = tf.trainable_variables()

        grads = tf.gradients(cost/tf.to_float(batch_size), trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op

    def inference(self, src_input):
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        with tf.variable_scope('encoder'):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, src_size, dtype=tf.float32)

        MAX_DEC_LEN = 100

        with tf.variable_scope('decoder/rnn/mnlti_rnn_cell'):
            init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
            init_array = init_array.write(0, SOS_ID)
            init_loop_var = (enc_state, init_array, 0)

            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(tf.not_equal(trg_ids.read(step), EOS_ID), tf.less(step, MAX_DEC_LEN-1)))

            def loop_body(state, trg_ids, step):
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

                dec_outputs, next_state = self.dec_cell.call(state=state, inputs=trg_emb)

                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = (tf.matmul(output, self.softmax_weight) + self.softmax_bias)

                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            state, trg_ids, step = tf.while_loop(continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


def run_epoch(session, cost_op, train_op, saver, step):
    while True:
        try:
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print('After %d steps, per token cost is %.3f' % (step, cost))
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def interence_main():
    preprocess = PreProcess()

    with tf.variable_scope('nmt_model', reuse=None):
        model = NMTModel()
        test_sentence = 'this is a test.'
        test_sentence = preprocess.english2id(test_sentence)
        output_op = model.inference(test_sentence)
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, CHECKPOINT_PATH)

        output = sess.run(output_op)

        print(output)
        print(preprocess.id2chinese(output))
        sess.close()


def main():
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope('nmt_model', reuse=None, initializer=initializer):
        train_model = NMTModel()

    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)

    saver = tf.train.Saver()
    step = 0

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCHS):
            print('In iteration: %d' % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == '__main__':
    main()

