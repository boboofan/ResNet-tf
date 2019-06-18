import tensorflow as tf


class ResNet18:
    def __init__(self, class_num, regularizer):
        self.class_num = class_num
        self.regularizer = regularizer

    def conv(self, inputs, output_dim, kernel_size=3, strides=1, padding='same'):
        return tf.layers.conv2d(inputs,
                                filters=output_dim,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                kernel_initializer=tf.truncated_normal_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))

    def res_block(self, inputs, output_dim, strides, training):
        input_dim = inputs.get_shape().as_list()[-1]

        if input_dim == output_dim:
            if strides == 2:
                shortcut = tf.layers.max_pooling2d(inputs, 2, 2, 'same')
            else:
                shortcut = inputs
        else:
            shortcut = tf.layers.conv2d(inputs, output_dim, 1, strides, 'same')

        conv1 = self.conv(inputs, output_dim, strides=strides)
        bn1 = tf.layers.batch_normalization(conv1, training=training)
        relu1 = tf.nn.relu(bn1)

        conv2 = self.conv(relu1, output_dim)
        bn2 = tf.layers.batch_normalization(conv2, training=training)
        relu2 = tf.nn.relu(bn2)

        return tf.add(shortcut, relu2)

    def output(self, inputs, training):  # [None,224,224,3]
        conv1 = self.conv(inputs, 64, 7, 2)
        conv1 = tf.layers.batch_normalization(conv1, training=training)
        conv1 = tf.nn.relu(conv1)

        conv2_1 = tf.layers.max_pooling2d(conv1, 3, 2, 'same')
        conv2_2 = self.res_block(conv2_1, 64, 1, training)
        conv2_3 = self.res_block(conv2_2, 64, 1, training)

        conv3_1 = self.res_block(conv2_3, 128, 2, training)
        conv3_2 = self.res_block(conv3_1, 128, 1, training)

        conv4_1 = self.res_block(conv3_2, 256, 2, training)
        conv4_2 = self.res_block(conv4_1, 256, 1, training)

        conv5_1 = self.res_block(conv4_2, 512, 2, training)
        conv5_2 = self.res_block(conv5_1, 512, 1, training)

        avg_pool = tf.layers.average_pooling2d(conv5_2, 7, 1, 'same')

        flatten = tf.keras.layers.Flatten()(avg_pool)

        fc1 = tf.layers.dense(flatten, 1000, tf.nn.relu,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))

        fc2 = tf.layers.dense(fc1, self.class_num)

        return fc2


class ResidualAttentionNet:
    def __init__(self, class_num, regularizer):
        self.class_num = class_num
        self.regularizer = regularizer

    def conv(self, inputs, output_dim, kernel_size=3, strides=1, padding='same'):
        return tf.layers.conv2d(inputs,
                                filters=output_dim,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))

    def res_unit(self, inputs, output_dim, strides, training):
        input_dim = inputs.get_shape().as_list()[-1]

        if input_dim == output_dim:
            if strides == 2:
                shortcut = tf.layers.average_pooling2d(inputs, 2, 2, 'same')
            else:
                shortcut = tf.identity(inputs)
        else:
            shortcut = self.conv(inputs, output_dim, strides=strides)

        conv1 = self.conv(inputs, output_dim, strides=strides)
        bn1 = tf.layers.batch_normalization(conv1, training=training)
        relu1 = tf.nn.relu(bn1)

        conv2 = self.conv(relu1, output_dim)
        bn2 = tf.layers.batch_normalization(conv2, training=training)
        relu2 = tf.nn.relu(bn2)

        return tf.add(shortcut, relu2)

    def mask_res_unit(self, inputs, training):
        input_dim = inputs.get_shape().as_list()[-1]

        bn1 = tf.layers.batch_normalization(inputs, training=training)
        relu1 = tf.nn.relu(bn1)
        conv1 = self.conv(relu1, input_dim)

        bn2 = tf.layers.batch_normalization(conv1, training=training)
        relu2 = tf.nn.relu(bn2)
        conv2 = self.conv(relu2, input_dim)

        return tf.add(inputs, conv2)

    def attention_module(self, inputs, n, training):
        input_dim = inputs.get_shape().as_list()[-1]

        # trunk branch
        trunk_branch = self.res_unit(inputs, input_dim, 1, training)
        trunk_branch = self.res_unit(trunk_branch, input_dim, 1, training)

        # mask branch
        mask_branch = inputs

        for i in range(n):
            mask_branch = tf.layers.max_pooling2d(mask_branch, 2, 2, 'same')
            mask_branch = self.mask_res_unit(mask_branch, training)

        for i in range(n):
            mask_branch = self.mask_res_unit(mask_branch, training)
            shape = mask_branch.get_shape().as_list()
            mask_branch = tf.image.resize_bilinear(mask_branch, [shape[1] * 2, shape[2] * 2])

        mask_branch = tf.nn.sigmoid(mask_branch)

        # H(x) = (1 + M(x)) ∗ F(x)
        attention_weight = tf.add(tf.ones_like(mask_branch), mask_branch)
        return tf.multiply(attention_weight, trunk_branch)

    def output(self, inputs, training):  # [None,224,224,3]
        with tf.name_scope('layer_1'):  # [None,112,112,64]
            conv = self.conv(inputs, 64, 7, 2)
            bn = tf.layers.batch_normalization(conv, training=training)
            relu = tf.nn.relu(bn)

        with tf.name_scope('layer_2'):  # [None,56,56,64]
            max_pool = tf.layers.max_pooling2d(relu, 3, 2, 'same')
            res1 = self.res_unit(max_pool, 64, 1, training)
            att1 = self.attention_module(res1, 3, training)

        with tf.name_scope('layer_3'):  # [None,28,28,128]
            res2 = self.res_unit(att1, 128, 2, training)
            att2 = self.attention_module(res2, 2, training)

        with tf.name_scope('layer_4'):  # [None,14,14,256]
            res3 = self.res_unit(att2, 256, 2, training)
            att3 = self.attention_module(res3, 1, training)

        with tf.name_scope('layer_5'):  # [None,7,7,512]
            res4 = self.res_unit(att3, 512, 2, training)
            res5 = self.res_unit(res4, 512, 1, training)

        with tf.name_scope('layer_6'):  # [None,1,1,512]
            avg_pool = tf.layers.average_pooling2d(res5, 7, 1, 'same')

        with tf.name_scope('fc'):
            flatten = tf.layers.flatten(avg_pool)
            fc = tf.layers.dense(flatten,
                                 units=1000,
                                 activation=tf.nn.relu,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularizer))
            return tf.layers.dense(fc,self.class_num)
