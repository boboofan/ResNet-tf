import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from resnet import ResidualAttentionNet


# tensorboard --logdir=D:/360data/重要数据/桌面/tf/logs
# http://localhost:6006

def get_dataset(data_path, train_size=0.8):
    img_paths, labels = [], []

    p=[]
    types = os.listdir(data_path)
    for i in range(len(types)):
        path = os.path.join(data_path, types[i])
        names = os.listdir(path)
        for name in names:
            if name.endswith(('.JPG', '.jpg', '.png', 'jpeg')):
                img_paths.append(os.path.join(path, name))
                labels.append(i)
    return train_test_split(img_paths, labels, train_size=train_size, random_state=1124)


def generator(X, Y, batch_size, training, img_size=(224, 224)):
    batch_start = 0
    while 1:
        batch_end = batch_start + batch_size if batch_start + batch_size < len(X) else len(X)
        xs = []
        for i in range(batch_start, batch_end):
            img = plt.imread(X[i])
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
            if training:
                img = np.pad(img, [[4, 4], [4, 4], [0, 0]], mode='constant')
                xx, yy = np.random.randint(5), np.random.randint(5)
                img = img[yy:yy + img_size[0], xx: xx + img_size[1]]
                flip = np.random.randint(4)
                if flip == 0:
                    img = img[::-1, :, :]
                elif flip == 1:
                    img = img[:, ::-1]
                elif flip == 2:
                    img = img[::-1, ::-1]
            xs.append(img)

        ys = Y[batch_start:batch_end]

        batch_start = batch_start + batch_size
        if batch_start >= len(X):
            batch_start = 0

        yield np.array(xs).astype(np.float32), np.array(ys).astype(np.int32)


def main():
    class_num = 38
    regularizer = 5e-4
    learning_rate = 0.005
    train_size = 43444
    test_size = 10861
    train_batch_size = 100
    test_batch_size = 100
    train_epochs = 20
    train_steps_each_epoch = int(np.ceil(train_size / train_batch_size))
    test_steps = int(np.ceil(test_size / test_batch_size))
    data_path = 'D:/360data/重要数据/桌面/项目/病虫害/raw/color'
    #data_path = 'Z:/Users/boboo/项目/病虫害/raw/color'
    model_save_path = './model_saving/'
    model_name = 'resattnet'
    log_path = './logs'

    train_x, test_x, train_y, test_y = get_dataset(data_path)
    print(len(train_y),len(test_y))

    train_iter = generator(train_x, train_y, train_batch_size, True)
    test_iter = generator(test_x, test_y, test_batch_size, False)

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(bool)

    model = ResidualAttentionNet(class_num, regularizer)
    logits = model.build_network(images, training)

    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
        tf.summary.scalar('mean_cross_entropy_loss', loss)

    with tf.name_scope('accuracy'):
        train_accuracy = tf.metrics.accuracy(labels, tf.argmax(tf.nn.softmax(logits), axis=-1, output_type=tf.int32))[1]
        tf.summary.scalar('train_accuracy', train_accuracy)

        test_accuracy = tf.metrics.accuracy(labels, tf.argmax(tf.nn.softmax(logits), axis=-1, output_type=tf.int32))[1]
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
        for epoch in range(train_epochs):
            for i in tqdm(range(train_steps_each_epoch)):
                xs, ys = next(train_iter)

                _, loss_val, acc_val, steps = sess.run([train_step, loss, train_accuracy, global_step],
                                                       feed_dict={images: xs, labels: ys, training: True})
                if i % 50 == 0:
                    print('epoch %d: after %d steps, loss is %.2f, accuracy is %.2f' % (epoch, i, loss_val, acc_val))

            saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)

            # test
            for i in tqdm(range(test_steps)):
                xs, ys = next(test_iter)

                acc_val = sess.run(test_accuracy, feed_dict={images: xs, labels: ys, training: False})
            print('epoch %d, accuracy on test dataset is %.2f' % (epoch, acc_val))


if __name__ == '__main__':
    main()
