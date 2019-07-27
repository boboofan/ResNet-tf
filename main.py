import tensorflow as tf
import resnet
import os
import matplotlib.pyplot as plt
from dataset_processing import image_resizing
from dataset_processing import get_dataset
from tqdm import tqdm

# tensorboard --logdir=D:/360data/重要数据/桌面/tf/logs
# http://localhost:6006


def generator(X, Y, batch_size, height, width):
    data_size = len(X)

    batch = 0
    while (1):
        xs = []
        for i in range(batch, batch + batch_size):
            img = plt.imread(X[i])
            xs.append(image_resizing(img, height, width))
        ys = Y[batch:batch + batch_size]

        batch = (batch + batch_size) % data_size

        yield (xs, ys)


def main():
    class_num = 38
    regularizer = 5e-4
    learning_rate_base = 0.01
    learning_rate_decay = 0.99
    train_size = 173700
    test_size = 43500
    train_batch_size = 100
    test_batch_size = 100
    width = 224
    height = 224
    epochs = 10
    root_path = 'Z:/Users/boboo/dataset/raw - copy/color'
    model_save_path = 'Z:/Users/boboo/tf/model/'
    model_name = 'resattnet'
    log_path = 'Z:/Users/boboo/tf/logs'

    train_x, test_x, train_y, test_y = get_dataset(root_path)

    train_iter = generator(train_x, train_y, train_batch_size, height, width)
    test_iter = generator(test_x, test_y, test_batch_size, height, width)

    inputs = tf.placeholder(tf.float32, [None, height, width, 3])
    labels = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(bool)

    model = resnet.ResidualAttentionNet(class_num, regularizer)
    outputs = model.output(inputs, training)

    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('learning_rate'):
        learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 10, learning_rate_decay,
                                                   staircase=True)
        tf.summary.scalar('exp_decay_lr', learning_rate)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=outputs))
        tf.summary.scalar('mean_cross_entropy_loss', loss)

    with tf.name_scope('accuracy'):
        train_accuracy = tf.metrics.accuracy(labels, tf.argmax(tf.nn.softmax(outputs), axis=-1, output_type=tf.int32))[1]
        tf.summary.scalar('train_accuracy', train_accuracy)
        test_accuracy = tf.metrics.accuracy(labels, tf.argmax(tf.nn.softmax(outputs), axis=-1, output_type=tf.int32))[1]
        tf.summary.scalar('test_accuracy', test_accuracy)
        
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # ckpt = tf.train.latest_checkpoint(model_save_path)
        # if ckpt:
        #    saver.restore(sess, ckpt)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        writer = tf.summary.FileWriter(log_path, sess.graph)
        merged = tf.summary.merge_all()

        # train
        for i in range(epochs):
            print('epoch %d:\n' % i)

            for j in tqdm(range(int(train_size / train_batch_size))):
                xs, ys = next(train_iter)

                _, loss_val, acc_val, lr, mer, steps = sess.run(
                    [train_step, loss, train_accuracy, learning_rate, merged, global_step],
                    feed_dict={inputs: xs, labels: ys, training: True})

                print('after %d steps, loss is %g, accuracy is %g, lr is %g' % (j, loss_val, acc_val, lr))

                if j % 100 == 0:
                    writer.add_summary(mer, steps)

            saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)

        # test
        for i in tqdm(range(int(test_size / test_batch_size))):
            xs, ys = next(test_iter)

            acc_val = sess.run(test_accuracy, feed_dict={inputs: xs, labels: ys, training: False})
            print('after %d steps, accuracy is %g' % (i, acc_val))


if __name__ == '__main__':
    main()
