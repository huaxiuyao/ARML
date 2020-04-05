import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

class LSTMAutoencoder(object):
    def __init__(self, hidden_num, input_num, cell=None, reverse=True, decode_without_input=False, name=None):
        self.name=name
        if cell is None:
            self._enc_cell = GRUCell(hidden_num, name='encoder_cell_{}'.format(self.name))
            self._dec_cell = GRUCell(hidden_num, name='decoder_cell_{}'.format(self.name))
        else:
            self._enc_cell = cell
            self._dec_cell = cell
        self.reverse = reverse
        self.decode_without_input = decode_without_input
        self.hidden_num = hidden_num

        if FLAGS.datasource in ['2D']:
            self.elem_num_init = 2
            self.elem_num=FLAGS.sync_filters

        elif FLAGS.datasource in ['plainmulti', 'artmulti']:
            self.elem_num = input_num

        self.dec_weight = tf.Variable(tf.truncated_normal([self.hidden_num,
                                                           self.elem_num], dtype=tf.float32), name='dec_weight_{}'.format(self.name))
        self.dec_bias = tf.Variable(tf.constant(0.1, shape=[self.elem_num],
                                                dtype=tf.float32), name='dec_bias_{}'.format(self.name))

    def model(self, inputs):

        inputs = tf.expand_dims(inputs, 0)

        inputs = tf.unstack(inputs, axis=1)

        self.batch_num = FLAGS.meta_batch_size

        with tf.variable_scope('encoder_{}'.format(self.name)):
            (self.z_codes, self.enc_state) = tf.contrib.rnn.static_rnn(self._enc_cell, inputs, dtype=tf.float32)

        with tf.variable_scope('decoder_{}'.format(self.name)) as vs:

            if self.decode_without_input:
                dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32) for _ in range(len(inputs))]
                (dec_outputs, dec_state) = tf.contrib.rnn.static_rnn(self._dec_cell, dec_inputs,
                                                                     initial_state=self.enc_state,
                                                                     dtype=tf.float32)
                if self.reverse:
                    dec_outputs = dec_outputs[::-1]
                dec_output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])
                dec_weight_ = tf.tile(tf.expand_dims(self.dec_weight, 0), [self.batch_num, 1, 1])
                self.output_ = tf.matmul(dec_weight_, dec_output_) + self.dec_bias
            else:
                dec_state = self.enc_state
                dec_input_ = tf.zeros(tf.shape(inputs[0]),
                                      dtype=tf.float32)

                dec_outputs = []
                for step in range(len(inputs)):
                    if step > 0:
                        vs.reuse_variables()
                    (dec_input_, dec_state) = \
                        self._dec_cell(dec_input_, dec_state)
                    dec_input_ = tf.matmul(dec_input_, self.dec_weight) + self.dec_bias
                    dec_outputs.append(dec_input_)
                if self.reverse:
                    dec_outputs = dec_outputs[::-1]
                self.output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])

        self.input_ = tf.transpose(tf.stack(inputs), [1, 0, 2])
        self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))

        self.emb_all = tf.reduce_mean(self.z_codes, axis=0)

        return self.emb_all, self.loss