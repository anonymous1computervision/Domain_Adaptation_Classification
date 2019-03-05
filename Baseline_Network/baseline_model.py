import sys

sys.path.append('../Data_Initialization/')
import tensorflow as tf
import tensorflow.contrib.layers as layers
import Initialization as init
import evaluation_function as eval
import time


class ResNet(object):
    def __init__(self, model_name, sess, train_data, val_data, tst_data, epoch, restore_epoch, num_class, ksize,
                 out_channel1, out_channel2, out_channel3, learning_rate, batch_size, img_height, img_width):

        self.sess = sess
        self.source_training_data = train_data[0]
        self.source_validation_data = val_data
        self.source_test_data = tst_data[0]
        self.target_training_data = train_data[1]
        self.target_test_data = tst_data[1]
        self.eps = epoch
        self.res_eps = restore_epoch
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.k = ksize
        self.oc1 = out_channel1
        self.oc2 = out_channel2
        self.oc3 = out_channel3
        self.lr = learning_rate
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.num_class = num_class

        self.build_model()
        self.saveConfiguration()

    def saveConfiguration(self):
        init.save2file('epoch : %d' % self.eps, self.ckptDir, self.model)
        init.save2file('restore epoch : %d' % self.res_eps, self.ckptDir, self.model)
        init.save2file('model : %s' % self.model, self.ckptDir, self.model)
        init.save2file('ksize : %d' % self.k, self.ckptDir, self.model)
        init.save2file('out channel 1 : %d' % self.oc1, self.ckptDir, self.model)
        init.save2file('out channel 2 : %d' % self.oc2, self.ckptDir, self.model)
        init.save2file('out channel 3 : %d' % self.oc3, self.ckptDir, self.model)
        init.save2file('learning rate : %g' % self.lr, self.ckptDir, self.model)
        init.save2file('batch size : %d' % self.bs, self.ckptDir, self.model)
        init.save2file('image height : %d' % self.img_h, self.ckptDir, self.model)
        init.save2file('image width : %d' % self.img_w, self.ckptDir, self.model)
        init.save2file('num class : %d' % self.num_class, self.ckptDir, self.model)

    def convLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_result = tf.layers.conv2d(inputs=inputMap, filters=out_channel, kernel_size=(ksize, ksize),
                                           strides=(stride, stride), padding=padding, use_bias=False,
                                           kernel_initializer=layers.variance_scaling_initializer(), name='conv')
            tf.summary.histogram('conv_result', conv_result)

            return conv_result

    def bnLayer(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name):
            return tf.layers.batch_normalization(inputMap, training=is_training, epsilon=1e-5)

    def reluLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.relu(inputMap)

    def avgPoolLayer(self, inputMap, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            return tf.nn.avg_pool(inputMap, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    def globalPoolLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            size = inputMap.get_shape()[1]
            return self.avgPoolLayer(inputMap, size, size, padding='VALID', scope_name=scope_name)

    def flattenLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.layers.flatten(inputMap)

    def fcLayer(self, inputMap, out_channel, scope_name):
        with tf.variable_scope(scope_name):
            fc_result = tf.layers.dense(inputs=inputMap, units=out_channel,
                                        kernel_initializer=layers.variance_scaling_initializer(), name='dense')

            tf.summary.histogram('fc_result', fc_result)

            return fc_result

    def residualUnitLayer(self, inputMap, out_channel, ksize, unit_name, down_sampling, is_training, first_conv=False):
        with tf.variable_scope(unit_name):
            in_channel = inputMap.get_shape().as_list()[-1]
            if down_sampling:
                stride = 2
                increase_dim = True
            else:
                stride = 1
                increase_dim = False

            if first_conv:
                conv_layer1 = self.convLayer(inputMap, out_channel, ksize, stride, scope_name='conv_layer1')
            else:
                bn_layer1 = self.bnLayer(inputMap, scope_name='bn_layer1', is_training=is_training)
                relu_layer1 = self.reluLayer(bn_layer1, scope_name='relu_layer1')
                conv_layer1 = self.convLayer(relu_layer1, out_channel, ksize, stride, scope_name='conv_layer1')

            bn_layer2 = self.bnLayer(conv_layer1, scope_name='bn_layer2', is_training=is_training)
            relu_layer2 = self.reluLayer(bn_layer2, scope_name='relu_layer2')
            conv_layer2 = self.convLayer(relu_layer2, out_channel, ksize, stride=1, scope_name='conv_layer2')

            if increase_dim:
                identical_mapping = self.avgPoolLayer(inputMap, ksize=2, stride=2, scope_name='identical_pool')
                identical_mapping = tf.pad(identical_mapping,
                                           [[0, 0], [0, 0], [0, 0],
                                            [(out_channel - in_channel) // 2, (out_channel - in_channel) // 2]],
                                           name='identical_padding')
            else:
                identical_mapping = inputMap

            added = tf.add(conv_layer2, identical_mapping)

            return added

    def residualStageLayer(self, inputMap, ksize, out_channel, unit_num, section_name, down_sampling, first_conv,
                           is_training):
        with tf.variable_scope(section_name):
            _out = inputMap
            _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_1', down_sampling=down_sampling,
                                          first_conv=first_conv, is_training=is_training)
            for n in range(2, unit_num + 1):
                _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_' + str(n),
                                              down_sampling=False, first_conv=False, is_training=is_training)

            return _out

    def resnet_model(self, input_x, model_name, unit_num1, unit_num2, unit_num3, reuse):
        with tf.variable_scope(model_name, reuse=reuse):
            _conv = self.convLayer(input_x, self.oc1, self.k, stride=1, scope_name='stage0_conv')
            _bn = self.bnLayer(_conv, scope_name='stage0_bn', is_training=self.is_training)
            _relu = self.reluLayer(_bn, scope_name='stage0_relu')

            stage1_out = self.residualStageLayer(inputMap=_relu,
                                                 ksize=self.k,
                                                 out_channel=self.oc1,
                                                 unit_num=unit_num1,
                                                 section_name='stage1',
                                                 down_sampling=False,
                                                 first_conv=True,
                                                 is_training=self.is_training)

            stage2_out = self.residualStageLayer(inputMap=stage1_out,
                                                 ksize=self.k,
                                                 out_channel=self.oc2,
                                                 unit_num=unit_num2,
                                                 section_name='stage2',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            stage3_out = self.residualStageLayer(inputMap=stage2_out,
                                                 ksize=self.k,
                                                 out_channel=self.oc3,
                                                 unit_num=unit_num3,
                                                 section_name='stage3',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            _trans_bn = self.bnLayer(stage3_out, scope_name='_trans_bn', is_training=self.is_training)
            _trans_relu = self.reluLayer(_trans_bn, scope_name='_trans_relu')
            _trans_gap = self.globalPoolLayer(_trans_relu, scope_name='_trans_gap')
            _trans_flatten = self.flattenLayer(_trans_gap, scope_name='_trans_flatten')

            y_pred = self.fcLayer(_trans_flatten, self.num_class, scope_name='fc')
            y_pred_softmax = tf.nn.softmax(y_pred)

            return y_pred, y_pred_softmax

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.img_h, self.img_w, 1])
        self.y = tf.placeholder(tf.int32, [None, self.num_class])
        self.is_training = tf.placeholder(tf.bool)
        tf.summary.image('Image/origin', self.x, max_outputs=3)

        self.y_pred, self.y_pred_softmax = self.resnet_model(input_x=self.x, model_name='classification_model',
                                                             unit_num1=3, unit_num2=3, unit_num3=3, reuse=False)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y))
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('tfSummary'):
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.ckptDir, self.sess.graph)

        with tf.variable_scope('saver'):
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.eps)

        with tf.variable_scope('accuracy'):
            self.distribution = [tf.argmax(self.y, 1), tf.argmax(self.y_pred_softmax, 1)]
            self.correct_prediction = tf.equal(self.distribution[0], self.distribution[1])
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'))

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.itr_epoch = len(self.source_training_data[0]) // self.bs
        self.total_iteration = self.eps * self.itr_epoch

        training_acc = 0.0
        training_loss = 0.0

        for itr in range(1, self.total_iteration + 1):
            _tr_img_batch, _tr_lab_batch = init.next_batch(image=self.source_training_data[0],
                                                           label=self.source_training_data[1],
                                                           batch_size=self.bs)

            _train_accuracy, _train_loss, _ = self.sess.run([self.accuracy, self.loss, self.train_op],
                                                            feed_dict={self.x: _tr_img_batch,
                                                                       self.y: _tr_lab_batch,
                                                                       self.is_training: True})
            training_acc += _train_accuracy
            training_loss += _train_loss

            if itr % self.itr_epoch == 0:
                _current_eps = int(itr / self.itr_epoch)
                summary = self.sess.run(self.merged, feed_dict={self.x: _tr_img_batch,
                                                                self.y: _tr_lab_batch,
                                                                self.is_training: False})

                training_acc = float(training_acc / self.itr_epoch)
                training_loss = float(training_loss / self.itr_epoch)

                validation_acc, validation_loss = eval.validation_procedure(validation_data=self.source_validation_data,
                                                                            distribution_op=self.distribution,
                                                                            loss_op=self.loss, inputX=self.x,
                                                                            inputY=self.y, num_class=self.num_class,
                                                                            batch_size=self.bs,
                                                                            is_training=self.is_training,
                                                                            session=self.sess)

                log = "Epoch: [%d], Training Accuracy: [%g], Validation Accuracy: [%g], Loss Training: [%g], " \
                      "Loss_validation: [%g], Time: [%s]" % \
                      (_current_eps, training_acc, validation_acc, training_loss, validation_loss,
                       time.ctime(time.time()))

                init.save2file(log, self.ckptDir, self.model)

                self.writer.add_summary(summary, _current_eps)

                self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(_current_eps))

                eval.test_procedure(test_data=self.source_test_data, distribution_op=self.distribution, inputX=self.x,
                                    inputY=self.y, mode='source', num_class=self.num_class, batch_size=self.bs,
                                    session=self.sess, is_training=self.is_training, ckptDir=self.ckptDir,
                                    model=self.model)

                eval.test_procedure(test_data=self.target_test_data, distribution_op=self.distribution, inputX=self.x,
                                    inputY=self.y, mode='target', num_class=self.num_class, batch_size=self.bs,
                                    session=self.sess, is_training=self.is_training, ckptDir=self.ckptDir,
                                    model=self.model)

                training_acc = 0.0
                training_loss = 0.0
