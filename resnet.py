import tensorflow as tf
import tensorflow.keras as keras


class Basic_Model:
    def __init__(self, regularizer, data_format='channels_last'):
        self.regularizer = regularizer
        self.data_format = data_format

    def conv2d(self, inputs, filters, kernel_size=3, strides=1, padding='same'):
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.regularizer),
                                      data_format=self.data_format)(inputs)

    def dense(self, inputs, units, activation=None):
        return tf.keras.layers.Dense(units=units, activation=activation,
                                     kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(inputs)

    def max_pool2d(self, inputs, pool_size=2, strides=2, padding='same'):
        return tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding,
                                         data_format=self.data_format)(inputs)

    def avg_pool2d(self, inputs, pool_size=2, strides=2, padding='same'):
        return tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding,
                                                data_format=self.data_format)(inputs)

    def global_avg_pool(self, inputs):
        return tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format)(inputs)

    def res_unit(self, inputs, output_dim, strides=1):
        input_dim = inputs.get_shape().as_list()[-1]

        if input_dim == output_dim:
            if strides == 2:
                shortcut = self.avg_pool2d(inputs)
            else:
                shortcut = tf.identity(inputs)
        else:
            shortcut = self.conv2d(inputs, filters=output_dim, kernel_size=1, strides=strides)

        conv1 = self.conv2d(inputs, output_dim, strides=strides)
        bn1 = tf.keras.layers.BatchNormalization()(conv1)
        relu1 = tf.keras.layers.ReLU()(bn1)

        conv2 = self.conv2d(relu1, output_dim)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        relu2 = tf.keras.layers.ReLU()(bn2)

        return tf.keras.layers.Add()([shortcut, relu2])


class CNN(Basic_Model):
    def build_network(self, inputs, classes_num):
        res1 = self.res_unit(inputs, output_dim=64, strides=2)
        res2 = self.res_unit(res1, output_dim=128, strides=2)
        avg = self.avg_pool2d(res2, pool_size=7, strides=1)

        gap = self.global_avg_pool(avg)
        fc = self.dense(gap, 1000, activation=tf.keras.layers.ReLU())

        return self.dense(fc, classes_num)


class Resnet18(Basic_Model):
    def build_network(self, inputs, classes_num):  # [None,224,224,3]
        conv1_1 = self.conv2d(inputs, filters=64, kernel_size=7, strides=2)
        bn = keras.layers.BatchNormalization()(conv1_1)
        relu = keras.layers.ReLU()(bn)

        max_pool = self.max_pool2d(relu, pool_size=3)
        conv2_1 = self.res_unit(max_pool, output_dim=64)
        conv2_2 = self.res_unit(conv2_1, output_dim=64)

        conv3_1 = self.res_unit(conv2_2, output_dim=128, strides=2)
        conv3_2 = self.res_unit(conv3_1, output_dim=128)

        conv4_1 = self.res_unit(conv3_2, output_dim=256, strides=2)
        conv4_2 = self.res_unit(conv4_1, output_dim=256)

        conv5_1 = self.res_unit(conv4_2, output_dim=512, strides=2)
        conv5_2 = self.res_unit(conv5_1, output_dim=512)

        avg_pool = self.avg_pool2d(conv5_2, pool_size=7, strides=1)

        gap = self.global_avg_pool(avg_pool)

        fc = self.dense(gap, 1000, activation=keras.layers.ReLU())

        return self.dense(fc, classes_num, activation=keras.layers.Softmax())


class ResnetAttention(Basic_Model):
    def conv_bn_activation(self, inputs, output_dim, kernel_size=3, strides=1, activation=None):
        conv = self.conv2d(inputs, filters=output_dim, kernel_size=kernel_size, strides=strides)
        bn = keras.layers.BatchNormalization()(conv)
        return activation(bn) if activation else bn

    def attention_module(self, inputs, n, p=1, t=2, r=1):
        output_dim = inputs.get_shape().as_list()[-1]

        head = inputs
        for i in range(p):
            head = self.res_unit(head, output_dim=output_dim)

        # trunk branch
        trunk_branch = head
        for i in range(t):
            trunk_branch = self.res_unit(trunk_branch, output_dim=output_dim)

        # mask branch
        mask_branch = head
        res_list = []

        for i in range(n):
            mask_branch = self.max_pool2d(mask_branch)
            for j in range(r):
                mask_branch = self.res_unit(mask_branch, output_dim=output_dim)
                if i < n - 1:
                    res_list.append(self.res_unit(mask_branch, output_dim=output_dim))

        for i in range(n):
            for j in range(r):
                mask_branch = self.res_unit(mask_branch, output_dim=output_dim)
            if tf.__version__ == '1.11.0':
                mask_branch = keras.backend.resize_images(mask_branch, 2, 2, data_format=self.data_format)
            else:
                mask_branch = keras.backend.resize_images(mask_branch, 2, 2, data_format=self.data_format,
                                                          interpolation='bilinear')

            if i < n - 1:
                mask_branch = keras.layers.Add()([mask_branch, res_list.pop()])

        mask_branch = self.conv_bn_activation(mask_branch, output_dim=output_dim, kernel_size=1,
                                              activation=keras.layers.ReLU())
        mask_branch = self.conv_bn_activation(mask_branch, output_dim=output_dim, kernel_size=1,
                                              activation=keras.layers.Activation('sigmoid'))

        # H(x) = (1 + M(x)) ∗ F(x)
        rear = keras.layers.Multiply()([(mask_branch + 1), trunk_branch])
        for i in range(p):
            rear = self.res_unit(rear, output_dim=output_dim)

        return rear

    def build_network(self, inputs, classes_num):
        layer1 = self.conv_bn_activation(inputs, output_dim=64, kernel_size=7, strides=2,
                                         activation=keras.layers.ReLU())

        max_pool = self.max_pool2d(layer1, pool_size=3)
        layer2 = self.res_unit(max_pool, output_dim=64)
        attention1 = self.attention_module(layer2, n=3)

        layer3 = self.res_unit(attention1, output_dim=128, strides=2)
        attention2 = self.attention_module(layer3, n=2)

        layer4 = self.res_unit(attention2, output_dim=256, strides=2)
        attention3 = self.attention_module(layer4, n=1)

        layer5 = self.res_unit(attention3, output_dim=512, strides=2)
        layer5 = self.res_unit(layer5, output_dim=512)

        avg_pool = self.avg_pool2d(layer5, pool_size=7, strides=1)

        gap = self.global_avg_pool(avg_pool)
        fc = self.dense(gap, units=1000, activation=keras.layers.ReLU())

        return self.dense(fc, units=classes_num, activation=keras.layers.Softmax())
