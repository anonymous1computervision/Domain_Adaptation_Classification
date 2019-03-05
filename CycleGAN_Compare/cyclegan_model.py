import sys

sys.path.append('../Data_Initialization/')
import tensorflow as tf
import tensorflow.contrib.layers as layers
import Initialization as init
import time
import evaluation_function as eval
import cyclegan_utils as utils
import numpy as np


class CycleGAN_Model(object):
    def __init__(self, model_name, sess, train_data, val_data, tst_data, epoch, num_class, learning_rate, batch_size,
                 img_height, img_width, train_phase, step, data_domain):

        self.sess = sess
        self.source_training_data = train_data[0]
        self.source_validation_data = val_data[0]
        self.source_test_data = tst_data[0]
        self.target_training_data = train_data[1]
        self.target_test_data = tst_data[1]
        self.eps = epoch
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.lr = learning_rate
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.num_class = num_class
        self.train_phase = train_phase
        self.step = step
        self.data_domain = data_domain

        if self.step == 1:
            self.build_cyclegan_model()
        if self.step == 2:
            self.build_classification_model()

        if self.train_phase == 'Train':
            self.saveConfiguration()

    def saveConfiguration(self):
        init.save2file('epoch : %d' % self.eps, self.ckptDir, self.model)
        init.save2file('model : %s' % self.model, self.ckptDir, self.model)
        init.save2file('learning rate : %g' % self.lr, self.ckptDir, self.model)
        init.save2file('batch size : %d' % self.bs, self.ckptDir, self.model)
        init.save2file('image height : %d' % self.img_h, self.ckptDir, self.model)
        init.save2file('image width : %d' % self.img_w, self.ckptDir, self.model)
        init.save2file('num class : %d' % self.num_class, self.ckptDir, self.model)
        init.save2file('train phase : %s' % self.train_phase, self.ckptDir, self.model)

    def convLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_result = tf.layers.conv2d(inputs=inputMap, filters=out_channel, kernel_size=(ksize, ksize),
                                           strides=(stride, stride), padding=padding, use_bias=False,
                                           kernel_initializer=layers.variance_scaling_initializer(), name='conv')
            tf.summary.histogram('conv_result', conv_result)

            return conv_result

    def convTransposeLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            deconv_result = tf.layers.conv2d_transpose(inputs=inputMap, filters=out_channel, kernel_size=(ksize, ksize),
                                                       strides=(stride, stride), padding=padding, use_bias=False,
                                                       kernel_initializer=layers.variance_scaling_initializer(),
                                                       name='deconv')

            tf.summary.histogram('deconv_result', deconv_result)

            return deconv_result

    def bnLayer(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name):
            return tf.layers.batch_normalization(inputMap, training=is_training)

    def reluLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.relu(inputMap)

    def lreluLayer(self, inputMap, scope_name, alpha=0.2):
        with tf.variable_scope(scope_name):
            return tf.nn.leaky_relu(inputMap, alpha=alpha)

    def avgpoolLayer(self, inputMap, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            return tf.nn.avg_pool(inputMap, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    def globalPoolLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            size = inputMap.get_shape()[1]
            return self.avgpoolLayer(inputMap, size, size, padding='VALID', scope_name=scope_name)

    def flattenLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.layers.flatten(inputMap)

    def fcLayer(self, inputMap, out_channel, scope_name):
        with tf.variable_scope(scope_name):
            fc_result = tf.layers.dense(inputs=inputMap, units=out_channel,
                                        kernel_initializer=layers.variance_scaling_initializer(), name='dense')

            tf.summary.histogram('fc_result', fc_result)

            return fc_result

    def convBnReluLayer(self, inputMap, ksize, stride, out_channel, scope_name, is_training, use_bn, use_relu):
        if use_relu:
            activation = self.reluLayer
        else:
            activation = self.lreluLayer

        with tf.variable_scope(scope_name):
            _conv = self.convLayer(inputMap, out_channel=out_channel, ksize=ksize, stride=stride,
                                   scope_name='_conv')
            if use_bn:
                _conv = self.bnLayer(_conv, scope_name='_bn', is_training=is_training)
            _relu = activation(_conv, scope_name='_relu')

        return _relu

    def Uet_G(self, inputMap, scope_name, is_training, reuse):
        with tf.variable_scope(scope_name, reuse=reuse):
            conv1_1 = self.convBnReluLayer(inputMap, ksize=3, stride=1, out_channel=64, scope_name='conv1_1',
                                           is_training=is_training, use_bn=True, use_relu=False)
            conv1_2 = self.convBnReluLayer(conv1_1, ksize=3, stride=1, out_channel=64, scope_name='conv1_2',
                                           is_training=is_training, use_bn=True, use_relu=False)

            conv2_1 = self.convBnReluLayer(conv1_2, ksize=3, stride=2, out_channel=128, scope_name='conv2_1',
                                           is_training=is_training, use_bn=True, use_relu=False)
            conv2_2 = self.convBnReluLayer(conv2_1, ksize=3, stride=1, out_channel=128, scope_name='conv2_2',
                                           is_training=is_training, use_bn=True, use_relu=False)

            conv3_1 = self.convBnReluLayer(conv2_2, ksize=3, stride=2, out_channel=256, scope_name='conv3_1',
                                           is_training=is_training, use_bn=True, use_relu=False)
            conv3_2 = self.convBnReluLayer(conv3_1, ksize=3, stride=1, out_channel=256, scope_name='conv3_2',
                                           is_training=is_training, use_bn=True, use_relu=False)

            up1 = self.convTransposeLayer(conv3_2, out_channel=128, ksize=3, stride=2, scope_name='up1')
            up1_bn = self.bnLayer(up1, scope_name='up1_bn', is_training=is_training)
            up1_cont = tf.concat([up1_bn, conv2_2], axis=3)
            up1_lrelu = self.lreluLayer(up1_cont, scope_name='up1_lrelu')

            up2 = self.convTransposeLayer(up1_lrelu, out_channel=64, ksize=3, stride=2, scope_name='up2')
            up2_bn = self.bnLayer(up2, scope_name='up2_bn', is_training=is_training)
            up2_cont = tf.concat([up2_bn, conv1_2], axis=3)
            up2_lrelu = self.lreluLayer(up2_cont, scope_name='up2_lrelu')

            conv_final = self.convLayer(up2_lrelu, out_channel=1, ksize=3, stride=1, scope_name='conv_final')
            conv_final_act = tf.nn.tanh(conv_final, name='conv_final_act')

        return conv_final_act

    def Discriminator(self, inputMap, ksize, scope_name, is_training, reuse):
        with tf.variable_scope(scope_name, reuse=reuse):
            _layer1 = self.convBnReluLayer(inputMap, out_channel=64, ksize=ksize, stride=1,
                                           scope_name='_layer1', is_training=is_training, use_bn=False, use_relu=False)
            _layer2 = self.convBnReluLayer(_layer1, out_channel=128, ksize=ksize, stride=2,
                                           scope_name='_layer2', is_training=is_training, use_bn=True, use_relu=False)
            _layer3 = self.convBnReluLayer(_layer2, out_channel=256, ksize=ksize, stride=2,
                                           scope_name='_layer3', is_training=is_training, use_bn=True, use_relu=False)
            _layer4 = self.convLayer(_layer3, out_channel=1, ksize=3, stride=1, scope_name='_layer4')

        return _layer4

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
                identical_mapping = self.avgpoolLayer(inputMap, ksize=2, stride=2, scope_name='identical_pool')
                identical_mapping = tf.pad(identical_mapping, [[0, 0], [0, 0], [0, 0],
                                                               [(out_channel - in_channel) // 2,
                                                                (out_channel - in_channel) // 2]])
            else:
                identical_mapping = inputMap

            added = tf.add(conv_layer2, identical_mapping)

            return added

    def residualSectionLayer(self, inputMap, ksize, out_channel, unit_num, section_name, down_sampling, first_conv,
                             is_training):
        with tf.variable_scope(section_name):
            _out = inputMap
            _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_1', down_sampling=down_sampling,
                                          first_conv=first_conv, is_training=is_training)
            for n in range(2, unit_num + 1):
                _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_' + str(n),
                                              down_sampling=False, first_conv=False, is_training=is_training)

            return _out

    def resnet_model(self, inputMap, model_name, ksize, unit_num1, unit_num2, unit_num3, out_channel1, out_channel2,
                     out_channel3, reuse):
        with tf.variable_scope(model_name, reuse=reuse):
            _conv = self.convLayer(inputMap, out_channel1, ksize=ksize, stride=1, scope_name='unit1_conv')
            _bn = self.bnLayer(_conv, scope_name='unit1_bn', is_training=self.is_training)
            _relu = self.reluLayer(_bn, scope_name='unit1_relu')

            sec1_out = self.residualSectionLayer(inputMap=_relu,
                                                 ksize=ksize,
                                                 out_channel=out_channel1,
                                                 unit_num=unit_num1,
                                                 section_name='section1',
                                                 down_sampling=False,
                                                 first_conv=True,
                                                 is_training=self.is_training)

            sec2_out = self.residualSectionLayer(inputMap=sec1_out,
                                                 ksize=ksize,
                                                 out_channel=out_channel2,
                                                 unit_num=unit_num2,
                                                 section_name='section2',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            sec3_out = self.residualSectionLayer(inputMap=sec2_out,
                                                 ksize=ksize,
                                                 out_channel=out_channel3,
                                                 unit_num=unit_num3,
                                                 section_name='section3',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            pred, pred_softmax = self.classifier(sec3_out, scope_name='classifier', reuse=reuse)

            return pred, pred_softmax

    def classifier(self, inputMap, scope_name, reuse):
        with tf.variable_scope(scope_name, reuse=reuse):
            _fm_bn = self.bnLayer(inputMap, scope_name='_fm_bn', is_training=self.is_training)
            _fm_relu = self.reluLayer(_fm_bn, scope_name='_fm_relu')
            _fm_pool = self.globalPoolLayer(_fm_relu, scope_name='_fm_gap')
            _fm_flatten = self.flattenLayer(_fm_pool, scope_name='_fm_flatten')

            y_pred = self.fcLayer(_fm_flatten, self.num_class, scope_name='fc_pred')
            y_pred_softmax = tf.nn.softmax(y_pred)

            return y_pred, y_pred_softmax

    def build_cyclegan_model(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='Y')

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.fake_y = self.Uet_G(inputMap=self.X, scope_name='G', is_training=self.is_training, reuse=False)
        self.fake_x = self.Uet_G(inputMap=self.Y, scope_name='F', is_training=self.is_training, reuse=False)

        self.rec_x = self.Uet_G(inputMap=self.fake_y, scope_name='F', is_training=self.is_training, reuse=True)
        self.rec_y = self.Uet_G(inputMap=self.fake_x, scope_name='G', is_training=self.is_training, reuse=True)

        self.real_y_dis = self.Discriminator(self.Y, ksize=3, scope_name='DY', is_training=self.is_training,
                                             reuse=False)
        self.fake_y_dis = self.Discriminator(self.fake_y, ksize=3, scope_name='DY', is_training=self.is_training,
                                             reuse=True)
        self.real_x_dis = self.Discriminator(self.X, ksize=3, scope_name='DX', is_training=self.is_training,
                                             reuse=False)
        self.fake_x_dis = self.Discriminator(self.fake_x, ksize=3, scope_name='DX', is_training=self.is_training,
                                             reuse=True)

        with tf.variable_scope('loss_functions'):
            # cycle_loss
            self.cycle_loss = 10 * tf.reduce_mean(tf.abs(self.rec_x - self.X)) + 10 * tf.reduce_mean(
                tf.abs(self.rec_y - self.Y))

            # X -> Y gan loss
            self.G_g_loss = tf.reduce_mean(tf.squared_difference(self.fake_y_dis, tf.ones_like(self.fake_y_dis)))
            self.G_loss = self.G_g_loss + self.cycle_loss
            self.DY_loss = (tf.reduce_mean(tf.squared_difference(self.real_y_dis, tf.ones_like(self.real_y_dis))) +
                            tf.reduce_mean(tf.squared_difference(self.fake_y_dis, tf.zeros_like(self.fake_y_dis)))) / 2

            # Y -> X gan loss
            self.F_g_loss = tf.reduce_mean(tf.squared_difference(self.fake_x_dis, tf.ones_like(self.fake_x_dis)))
            self.F_loss = self.F_g_loss + self.cycle_loss
            self.DX_loss = (tf.reduce_mean(tf.squared_difference(self.real_x_dis, tf.ones_like(self.real_x_dis))) +
                            tf.reduce_mean(tf.squared_difference(self.fake_x_dis, tf.zeros_like(self.fake_x_dis)))) / 2

        tf.summary.scalar('loss/G', self.G_g_loss)
        tf.summary.scalar('loss/DY', self.DY_loss)
        tf.summary.scalar('loss/F', self.F_g_loss)
        tf.summary.scalar('loss/DX', self.DX_loss)
        tf.summary.scalar('loss/cycle', self.cycle_loss)

        tf.summary.image('X/origin', self.X)
        tf.summary.image('Y/origin', self.Y)
        tf.summary.image('X/generated', self.fake_x)
        tf.summary.image('X/reconstructed', self.rec_x)
        tf.summary.image('Y/generated', self.fake_y)
        tf.summary.image('Y/reconstructed', self.rec_y)

        with tf.variable_scope('optimization_variables'):
            self.G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
            self.F_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='F')

            self.DX_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DX')
            self.DY_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DY')

        with tf.variable_scope('optimize'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')):
                self.G_trainOp = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.G_loss,
                                                                                     var_list=self.G_var)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='F')):
                self.F_trainOp = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.F_loss,
                                                                                     var_list=self.F_var)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='DX')):
                self.DX_trainOp = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.DX_loss,
                                                                                      var_list=self.DX_var)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='DY')):
                self.DY_trainOp = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.DY_loss,
                                                                                      var_list=self.DY_var)

        with tf.variable_scope('tfSummary'):
            self.merged = tf.summary.merge_all()
            if self.train_phase == 'Train':
                self.writer = tf.summary.FileWriter(self.ckptDir, self.sess.graph)

        with tf.variable_scope('saver'):
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.eps)

    def build_classification_model(self):
        self.cla_x = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='cla_x')
        self.direct_input_x = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='direct_input_x')
        self.cla_y = tf.placeholder(tf.int32, shape=[None, self.num_class], name='cla_y')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        if self.data_domain == 'Source':
            self.reload_g_name = 'G'
        elif self.data_domain == 'Target':
            self.reload_g_name = 'F'
        else:
            print('Wrong Data Domain')
            exit(3)

        self.fake_x = self.Uet_G(inputMap=self.cla_x, scope_name=self.reload_g_name, is_training=self.is_training,
                                 reuse=False)
        tf.summary.image('Image/origin_x', self.cla_x)
        tf.summary.image('Image/fake_x', self.fake_x)

        self.pred, self.pred_softmax = self.resnet_model(inputMap=self.fake_x,
                                                         model_name='classification_model',
                                                         ksize=3,
                                                         unit_num1=3,
                                                         unit_num2=3,
                                                         unit_num3=3,
                                                         out_channel1=16,
                                                         out_channel2=32,
                                                         out_channel3=64,
                                                         reuse=False)

        self.pred_tar_tst, self.pred_softmax_tar_tst = self.resnet_model(inputMap=self.direct_input_x,
                                                                         model_name='classification_model',
                                                                         ksize=3,
                                                                         unit_num1=3,
                                                                         unit_num2=3,
                                                                         unit_num3=3,
                                                                         out_channel1=16,
                                                                         out_channel2=32,
                                                                         out_channel3=64,
                                                                         reuse=True)

        with tf.variable_scope('loss_functions'):
            # supervised loss
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.cla_y))
            tf.summary.scalar('supervised_loss', self.loss)

        with tf.variable_scope('optimization_variables'):
            self.cla_model_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classification_model')
            self.reload_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.reload_g_name)

        with tf.variable_scope('optimize'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='classification_model')):
                self.classification_model_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                                                              var_list=self.cla_model_var)

        with tf.variable_scope('tfSummary'):
            self.merged = tf.summary.merge_all()
            if self.train_phase == 'Train':
                self.writer = tf.summary.FileWriter(self.ckptDir, self.sess.graph)

        with tf.variable_scope('saver'):
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.eps)

        with tf.variable_scope('accuracy'):
            self.distribution = [tf.argmax(self.cla_y, 1), tf.argmax(self.pred_softmax, 1)]
            self.correct_prediction = tf.equal(self.distribution[0], self.distribution[1])
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'))

            self.distribution_tar_tst = [tf.argmax(self.cla_y, 1), tf.argmax(self.pred_softmax_tar_tst, 1)]
            self.correct_prediction_tar_tst = tf.equal(self.distribution_tar_tst[0], self.distribution_tar_tst[1])
            self.accuracy_tar_tst = tf.reduce_mean(tf.cast(self.correct_prediction_tar_tst, 'float'))

    def getBatchData_1(self):
        _src_tr_img_batch, _src_tr_lab_batch = init.next_batch(self.source_training_data[0],
                                                               self.source_training_data[1], self.bs)
        _tar_tr_img_batch = init.next_batch_unpaired(self.target_training_data, self.bs)

        feed_dict = {self.X: _src_tr_img_batch,
                     self.Y: _tar_tr_img_batch,
                     self.is_training: True}

        feed_dict_eval = {self.X: _src_tr_img_batch,
                          self.Y: _tar_tr_img_batch,
                          self.is_training: False}

        return feed_dict, feed_dict_eval

    def getBatchData_2(self):
        _src_tr_img_batch, _src_tr_lab_batch = init.next_batch(self.source_training_data[0],
                                                               self.source_training_data[1], self.bs)

        feed_dict = {self.cla_x: _src_tr_img_batch,
                     self.direct_input_x: _src_tr_img_batch,
                     self.cla_y: _src_tr_lab_batch,
                     self.is_training: True}

        feed_dict_eval = {self.cla_x: _src_tr_img_batch,
                          self.direct_input_x: _src_tr_img_batch,
                          self.cla_y: _src_tr_lab_batch,
                          self.is_training: False}

        return feed_dict, feed_dict_eval

    def train_step1(self):
        self.sess.run(tf.global_variables_initializer())
        self.itr_epoch = len(self.source_training_data[0]) // self.bs

        for e in range(1, self.eps + 1):
            for itr in range(self.itr_epoch):
                feed_dict_train, feed_dict_eval = self.getBatchData_1()
                _, _, _, _ = self.sess.run([self.G_trainOp, self.F_trainOp, self.DX_trainOp, self.DY_trainOp],
                                           feed_dict=feed_dict_train)

            summary = self.sess.run(self.merged, feed_dict=feed_dict_eval)

            G_g_loss, F_g_loss, DX_loss, DY_loss, Cycle_loss = self.sess.run(
                [self.G_g_loss, self.F_g_loss, self.DX_loss, self.DY_loss, self.cycle_loss], feed_dict=feed_dict_eval)

            log1 = "Epoch: [%d], G_g_loss: [%g], F_g_loss: [%g], DX_loss: [%g], DY_loss: [%g], Cycle_loss: [%g], " \
                   "Time: [%s]" % (e, G_g_loss, F_g_loss, DX_loss, DY_loss, Cycle_loss, time.ctime(time.time()))

            init.save2file(log1, self.ckptDir, self.model)

            self.writer.add_summary(summary, e)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e))

    def test_procedure(self, test_data, distribution_op, inputX, inputX_, inputY, mode, num_class, batch_size, session,
                       is_training, ckptDir, model):
        confusion_matrics = np.zeros([num_class, num_class], dtype="int")

        tst_batch_num = int(np.ceil(test_data[0].shape[0] / batch_size))
        for step in range(tst_batch_num):
            _testImg = test_data[0][step * batch_size:step * batch_size + batch_size]
            _testLab = test_data[1][step * batch_size:step * batch_size + batch_size]

            matrix_row, matrix_col = session.run(distribution_op, feed_dict={inputX: _testImg,
                                                                             inputX_: _testImg,
                                                                             inputY: _testLab,
                                                                             is_training: False})
            for m, n in zip(matrix_row, matrix_col):
                confusion_matrics[m][n] += 1

        test_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(num_class)])) / float(
            np.sum(confusion_matrics))
        detail_test_accuracy = [confusion_matrics[i][i] / np.sum(confusion_matrics[i]) for i in
                                range(num_class)]
        log0 = "Mode: " + mode
        log1 = "Test Accuracy : %g" % test_accuracy
        log2 = np.array(confusion_matrics.tolist())
        log3 = ''
        for j in range(num_class):
            log3 += 'category %s test accuracy : %g\n' % (init.pulmonary_category[j], detail_test_accuracy[j])
        log3 = log3[:-1]
        log4 = 'F_Value : %g\n' % eval.f_value(confusion_matrics, num_class)

        init.save2file(log0, ckptDir, model)
        init.save2file(log1, ckptDir, model)
        init.save2file(log2, ckptDir, model)
        init.save2file(log3, ckptDir, model)
        init.save2file(log4, ckptDir, model)

    def train_step2(self, reload_path):
        self.plt_epoch = []
        self.plt_training_accuracy = []
        self.plt_training_loss = []

        # 全局初始化
        self.sess.run(tf.global_variables_initializer())
        print('Global Initialization Finish')

        # 生成器参数重载
        reload_saver = tf.train.Saver(var_list=self.reload_var)
        reload_saver.restore(self.sess, reload_path)
        print('Generator %s Reload Finish' % self.reload_g_name)

        # 开始训练
        self.itr_epoch = len(self.source_training_data[0]) // self.bs
        training_acc = 0.0
        training_loss = 0.0

        for e in range(1, self.eps + 1):
            for itr in range(self.itr_epoch):
                feed_dict_train, feed_dict_eval = self.getBatchData_2()
                _ = self.sess.run(self.classification_model_train_op, feed_dict=feed_dict_train)

                _training_accuracy, _training_loss = self.sess.run([self.accuracy, self.loss], feed_dict=feed_dict_eval)
                training_acc += _training_accuracy
                training_loss += _training_loss

            summary = self.sess.run(self.merged, feed_dict=feed_dict_eval)

            training_acc = float(training_acc / self.itr_epoch)
            training_loss = float(training_loss / self.itr_epoch)

            self.plt_epoch.append(e)
            self.plt_training_accuracy.append(training_acc)
            self.plt_training_loss.append(training_loss)

            utils.plotAccuracy(x=self.plt_epoch,
                               y1=self.plt_training_accuracy,
                               figName=self.model,
                               line1Name='training',
                               savePath=self.ckptDir,
                               y2=None,
                               line2Name='')
            utils.plotLoss(x=self.plt_epoch,
                           y1=self.plt_training_loss,
                           figName=self.model,
                           line1Name='loss',
                           savePath=self.ckptDir,
                           y2=None,
                           line2Name='')

            log1 = "Epoch: [%d], Training accuracy: [%g], Training loss: [%g], Time: [%s]" % (
                e, training_acc, training_loss, time.ctime(time.time()))

            init.save2file(log1, self.ckptDir, self.model)

            self.writer.add_summary(summary, e)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e))

            self.test_procedure(self.source_test_data, distribution_op=self.distribution_tar_tst,
                                inputX=self.direct_input_x, inputX_=self.cla_x, inputY=self.cla_y, mode='source',
                                num_class=self.num_class, batch_size=self.bs, session=self.sess,
                                is_training=self.is_training, ckptDir=self.ckptDir, model=self.model)

            self.test_procedure(self.target_test_data, distribution_op=self.distribution_tar_tst,
                                inputX=self.direct_input_x, inputX_=self.cla_x, inputY=self.cla_y, mode='target',
                                num_class=self.num_class, batch_size=self.bs, session=self.sess,
                                is_training=self.is_training, ckptDir=self.ckptDir, model=self.model)
