import ipdb
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

class GraphConvolution(object):

    def __init__(self, hidden_dim, name=None, sparse_inputs=False, act=tf.nn.tanh, bias=True, dropout=0.0):
        self.act = act
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias

        with tf.variable_scope('{}_vars'.format(name), tf.AUTO_REUSE):
            self.gcn_weights = tf.Variable(tf.truncated_normal([self.hidden_dim, self.hidden_dim], dtype=tf.float32),
                                           name='gcn_weight')
            if self.bias:
                self.gcn_bias = tf.Variable(tf.constant(0.0, shape=[self.hidden_dim],
                                                        dtype=tf.float32), name='gcn_bias')

    def model(self, feat, adj):
        x = feat
        x = tf.nn.dropout(x, 1 - self.dropout)

        node_size = tf.shape(adj)[0]
        I = tf.eye(node_size)
        adj = adj + I
        D = tf.diag(tf.reduce_sum(adj, axis=1))
        adj = tf.matmul(tf.linalg.inv(D), adj)
        pre_sup = tf.matmul(x, self.gcn_weights)
        output = tf.matmul(adj, pre_sup)

        if self.bias:
            output += self.gcn_bias
        if self.act is not None:
            return self.act(output)
        else:
            return output

class MetaGraph(object):
    def __init__(self, hidden_dim, input_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proto_num = FLAGS.num_classes
        self.node_cluster_center, self.nodes_cluster_bias = [], []
        for i in range(FLAGS.num_vertex):
            self.node_cluster_center.append(tf.get_variable(name='{}_node_cluster_center'.format(i),
                                                            shape=(1, input_dim)))
            self.nodes_cluster_bias.append(
                tf.get_variable(name='{}_nodes_cluster_bias'.format(i), shape=(1, hidden_dim)))

        self.vertex_num = FLAGS.num_vertex

        self.adj_mlp_weight = tf.Variable(tf.truncated_normal([self.hidden_dim, 1], dtype=tf.float32),
                                          name='adj_mlp_weight')
        self.adj_mlp_bias = tf.Variable(tf.constant(0.1, shape=[1],
                                                    dtype=tf.float32), name='adj_mlp_bias')

        self.GCN = GraphConvolution(self.hidden_dim, name='data_gcn')

    def model(self, inputs):
        if FLAGS.datasource in ['plainmulti', 'artmulti']:
            sigma = 8.0
        elif FLAGS.datasource in ['2D']:
            sigma = 2.0

        cross_graph = tf.nn.softmax(
            (-tf.reduce_sum(tf.square(inputs - self.node_cluster_center), axis=-1) / (2.0 * sigma)), axis=0)
        cross_graph = tf.transpose(cross_graph, perm=[1, 0])

        meta_graph = []
        for idx_i in range(self.vertex_num):
            tmp_dist = []
            for idx_j in range(self.vertex_num):
                if idx_i == idx_j:
                    dist = tf.squeeze(tf.zeros([1]))
                else:
                    dist = tf.squeeze(tf.sigmoid(tf.layers.dense(
                        tf.abs(self.node_cluster_center[idx_i] - self.node_cluster_center[idx_j]), units=1,
                        name='meta_dist')))
                tmp_dist.append(dist)
            meta_graph.append(tf.stack(tmp_dist))
        meta_graph = tf.stack(meta_graph)

        proto_graph = []
        for idx_i in range(self.proto_num):
            tmp_dist = []
            for idx_j in range(self.proto_num):
                if idx_i == idx_j:
                    dist = tf.squeeze(tf.zeros([1]))
                else:
                    dist = tf.squeeze(tf.sigmoid(tf.layers.dense(
                        tf.abs(tf.expand_dims(inputs[idx_i] - inputs[idx_j], axis=0)), units=1, name='proto_dist')))
                tmp_dist.append(dist)
            proto_graph.append(tf.stack(tmp_dist))
        proto_graph = tf.stack(proto_graph)

        adj = tf.concat((tf.concat((proto_graph, cross_graph), axis=1),
                         tf.concat((tf.transpose(cross_graph, perm=[1, 0]), meta_graph), axis=1)), axis=0)

        feat = tf.concat((inputs, tf.squeeze(tf.stack(self.node_cluster_center))), axis=0)

        repr = self.GCN.model(feat, adj)

        return repr[0:self.proto_num]
