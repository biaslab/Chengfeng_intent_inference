import tensorflow.compat.v1 as tf
from get_data import get_data,gettest_data
tf.disable_v2_behavior() 
# from tensorflow.examples.tutorials.mnist import input_data
 
 
# 导入数据
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
 
# hyperparameters 超参数
lr = 0.001
training_iters = 10000
# batch_size = 63860
 
n_inputs = 8  # (img shape:28 * 28)
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
 
 
def RNN(X, weights, biases,batch_size):
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
batch_size=1
pred = RNN(x, weights, biases,batch_size)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
pre_label=tf.argmax(pred, 1)[0]+1
true_label=tf.argmax(y, 1)[0]+1
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
dic=gettest_data()
path=r"/Users/jia/Documents/intent_inference/compare_model/elstm_res/5/"

init = tf.initialize_all_variables()
saver = tf.train.Saver()


with tf.Session() as sess:
    saver.restore(sess, "/Users/jia/Documents/intent_inference/compare_model/model_elstm/lstm5/model.ckpt")
    for key in dic.keys():
        
        data=dic[key]["data"]
        label=dic[key]["label"]
        
        fw=open(path+"own_res_%s.csv"%(key),"w")
        for li in range(len(data)):
            pre,true=sess.run([pre_label,true_label], feed_dict={
                        x: [data[li]],
                        y: [label[li]],
                    })
            # print(pre,true)
            fw.write(str(pre)+","+str(true)+"\n")
        # break
    # tf.reset_default_graph()
