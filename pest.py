import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from resnet import Resnet18, ResnetAttention


# tensorboard --logdir=D:/360data/重要数据/桌面/tf/logs
# http://localhost:6006

def get_dataset(data_path, train_size=0.8):
    img_paths, labels = [], []

    types = os.listdir(data_path)
    for i in range(len(types)):
        path = os.path.join(data_path, types[i])
        names = os.listdir(path)
        for name in names:
            if name.endswith(('.JPG', '.jpg', '.png', 'jpeg')):
                img_paths.append(os.path.join(path, name))
                labels.append(i)
    return train_test_split(img_paths, labels, train_size=train_size)


def data_iter(X, Y, batch_size, training, img_size=(224, 224)):
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
    classes_num = 4
    regularizer = 5e-4
    learning_rate = 0.005
    epochs = 30
    train_size = 2536
    test_size = 635
    train_batch_size = 10
    test_batch_size = 10
    train_steps = int(np.ceil(train_size / train_batch_size))
    test_steps = int(np.ceil(test_size / test_batch_size))
    data_path = 'D:/360data/重要数据/桌面/tf-keras/images'
    #data_path = '/home/boboo/pest/ResNet-tf/images'

    train_x, test_x, train_y, test_y = get_dataset(data_path)
    train_iter = data_iter(train_x, train_y, train_batch_size, True)
    test_iter = data_iter(test_x, test_y, test_batch_size, False)

    images = keras.Input([224, 224, 3], dtype=tf.float32)
    resnet_attention = ResnetAttention(regularizer)
    outputs = resnet_attention.build_network(images, classes_num=classes_num)

    model = keras.Model(images, outputs)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # tf.keras.backend.set_session(sess)

    model.fit_generator(generator=train_iter, steps_per_epoch=train_steps, epochs=epochs,
                        validation_data=test_iter, validation_steps=test_steps)


if __name__ == '__main__':
    main()
