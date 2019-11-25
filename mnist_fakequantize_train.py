# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_build_network import build_network, x, y, keep_prob
from tensorflow.contrib.quantize import *

# 加载mnist下载的数据，有四个，数据我是在一个博客给出的百度网盘里下的，不过我的git账号还没弄好，不能上传，需要的话可以去看mnist实现的相关博客找数据下载。
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def create_training_graph():
#创建训练图，加入create_training_graph：
    g = tf.get_default_graph()   # 给create_training_graph的参数，默认图 
#调用网络定义，也就是拿到输出
    logits = build_network(is_training=True)    #这里的is_training设置为True，因为前面模型定义写了训练时要用到dropout
# 写loss，mnist的loss是用交叉熵来计算的，loss和optimize方法可以根据自己的情况来设置。
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        print('cost:', cross_entropy_mean)
# 加入 create_training_graph函数，注意位置要在loss之后， optimize之前
    # if FLAGS.quantize:     
    # 上面这句是如果用parser设置flag参数的话，就用这种方式设置开关，用法可以自己查一下，或者参考speech的例子就知道了。
    tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=0)
#  optimize用原来的Adam效果较好，不知道我这里为什么用GradientDescentOptimizer的话，基本不收敛。
    optimize = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean)
    # optimize = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy_mean)
# 比较输出类别概率的最大值[tf.argmax 是项的极其有益的函数，它给返回在一个标题里最大值的索引。例如，tf.argmax(y,1) 是我们的模型输出的认为是最有可能是的那个值，而 tf.argmax(y_,1) 是正确的标签的标签。]
    #prediction_labels = tf.argmax(logits, axis=1, name="output")
# 将得出的最大值与实际分类标签对比，看二者是否一致[如果我们的预测与匹配真正的值，我们可以使用tf.equal来检查。]    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# 给出识别准确率[这会返回我们一个布尔值的列表.为了确定哪些部分是正确的，我们要把它转换成浮点值，然后再示均值。 比如, [True, False, True, True] 会转换成 [1,0,1,1] ，从而它的准确率就是0.75.]    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#后面with这部分对于量化应该没啥影响，记得是tensorboard要用的，应该是出准确率啥的曲线图的……（不过这里应该没用到吧，，我只是搬过来了还没看，会用的想用就用吧，不会的就删掉吧）
    with tf.get_default_graph().name_scope('eval'):
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        tf.summary.scalar('accuracy', accuracy)
# 返回所需数据，供训练使用        
    return dict(
        x=x,
        y=y,
        keep_prob=keep_prob,
        optimize=optimize,
        cost=cross_entropy_mean,
        correct_prediction=correct_prediction,
        accuracy=accuracy,
    )
    
#开始训练
def train_network(graph):
# 初始化
    init = tf.global_variables_initializer()
    # 调用Saver函数保存所需文件
    saver = tf.train.Saver()
    # 创建上下文，开始训练sess.run(init)
    with tf.Session() as sess:
        sess.run(init)
        # 一共训练两万次，也可以更多，不过两万次感觉准确率就能达到将近1了
        for i in range(20000):
        # 每次处理50张图片
            batch = mnist.train.next_batch(50)
            # 每100次保存并打印一次准确率等
            if i % 100 == 0:
            # feed_dict喂数据
                train_accuracy = sess.run([graph['accuracy']], feed_dict={
                                                                           graph['x']:batch[0],    # batch[0]存的图片数据
                                                                           graph['y']:batch[1],    # batch[1]存的标签
                                                                           graph['keep_prob']: 1.0})      # 随机失活(全部？)
                print("step %d, training accuracy %g"%(i, train_accuracy[0]))
            sess.run([graph['optimize']], feed_dict={
                                                       graph['x']:batch[0],
                                                       graph['y']:batch[1],
                                                       graph['keep_prob']:0.5})
        test_accuracy = sess.run([graph['accuracy']], feed_dict={
                                                                  graph['x']: mnist.test.images,
                                                                  graph['y']: mnist.test.labels,
                                                                  graph['keep_prob']: 1.0})
        print("Test accuracy %g" % test_accuracy[0])
# 保存ckpt(checkpoint)和pbtxt。记得把路径改成自己的路径，写不好相对路径的就直接写绝对路径。绝对路径就是我写的这种完整的路径。
        saver.save(sess, '/MNIST/ckpt/mnist_fakequantize.ckpt')
        tf.train.write_graph(sess.graph_def, '/MNIST/ckpt', 'mnist_fakequantize.pbtxt', True)

def main():
    g1 = create_training_graph()
    train_network(g1)

main()

