#encoding:utf-8
import tensorflow.compat.v1 as tf
from get_data import get_data
tf.disable_v2_behavior() 
 
 
# hyperparameters 超参数
lr = 0.001
training_iters = 10000
batch_size = 50
 
n_inputs = 8  #
n_steps = 5  # time steps
n_hidden_units = 64  # 隐藏层神经元数目
n_classes = 9  # classes(0-9 digits)
 
# 输入
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
 
# 定义权值
 
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}
 
 
def RNN(X, weights, biases):
    # 隐藏层输入到cell
    # X(128 batch, 28 steps, 28 inputs)
    #  ==>(128*28,28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in==>(128batch*28steps, 128 hidden)
    X_in = tf.matmul(X, weights['in'] + biases['in'])
    # X_in==>(128batch, 28steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell 被分成两部分，(c_state, m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    # 隐藏层输出
    # 第一种方式
    results = tf.matmul(states[1], weights['out']) + biases['out']
    # 第二种方式
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  # state is the last outputs
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results
 
 
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
data,label=get_data()
print(data.shape)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    ii=0
    while step * batch_size < training_iters:
        if ii+batch_size>len(data):
            ii=0
        batch_xs, batch_ys = data[ii:ii+batch_size,:,:],label[ii:ii+batch_size,:]
        ii=ii+batch_size
        # batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
        step += 1
        
        saver.save(sess, "/Users/jia/Documents/intent_inference/compare_model/model_lstm/model.ckpt")

