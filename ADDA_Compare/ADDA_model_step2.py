import sys

sys.path.append('../Data_Initialization/')
import tensorflow as tf
import tensorflow.contrib.layers as layers
import Initialization as init
import ADDA_utils as utils
import evaluation_function as eval
import time


class DA_Model_step2(object):
    def __init__(self, model_name, sess, train_data, val_data, tst_data, epoch, reloadPath, num_class, learning_rate,
                 batch_size, img_height, img_width, train_phase, step):

        self.sess = sess
        self.source_training_data = train_data[0]
        self.source_validation_data = val_data[0]
        self.source_test_data = tst_data[0]
        self.target_training_data = train_data[1]
        self.target_test_data = tst_data[1]
        self.eps = epoch
        self.reloadPath = reloadPath
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.lr = learning_rate
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.num_class = num_class
        self.train_phase = train_phase
        self.step = step
        self.plt_epoch = []
        self.plt_training_accuracy = []
        self.plt_g_loss = []
        self.plt_d_loss = []

        self.build_model()
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
        init.save2file('step : %d' % self.step, self.ckptDir, self.model)

    def convLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_result = tf.layers.conv2d(inputs=inputMap, filters=out_channel, kernel_size=(ksize, ksize),
                                           strides=(stride, stride), padding=padding, use_bias=False,
                                           kernel_initializer=layers.variance_scaling_initializer(), name='conv')
            tf.summary.histogram('conv_result', conv_result)

            return conv_result

    def bnLayer(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name):
            return tf.layers.batch_normalization(inputMap, training=is_training)

    def reluLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.relu(inputMap)

    def lreluLayer(self, inputMap, scope_name, alpha=0.2):
        with tf.variable_scope(scope_name):
            return tf.nn.leaky_relu(inputMap, alpha=alpha)

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

    def convBnReluLayer(self, inputMap, ksize, stride, out_channel, scope_name, is_training, use_bn):
        with tf.variable_scope(scope_name):
            _conv = self.convLayer(inputMap, out_channel=out_channel, ksize=ksize, stride=stride,
                                   scope_name='_conv')
            if use_bn:
                _conv = self.bnLayer(_conv, scope_name='_bn', is_training=is_training)
            _relu = self.lreluLayer(_conv, scope_name='_relu')

        return _relu

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

            return sec3_out

    def classifier(self, inputMap, scope_name, reuse):
        with tf.variable_scope(scope_name, reuse=reuse):
            _fm_bn = self.bnLayer(inputMap, scope_name='_fm_bn', is_training=self.is_training)
            _fm_relu = self.reluLayer(_fm_bn, scope_name='_fm_relu')
            _fm_pool = self.globalPoolLayer(_fm_relu, scope_name='_fm_gap')
            _fm_flatten = self.flattenLayer(_fm_pool, scope_name='_fm_flatten')

            y_pred = self.fcLayer(_fm_flatten, self.num_class, scope_name='fc_pred')
            y_pred_softmax = tf.nn.softmax(y_pred)

            return y_pred, y_pred_softmax

    def Discriminator(self, inputMap, ksize, scope_name, is_training, reuse):
        with tf.variable_scope(scope_name, reuse=reuse):
            in_channel = inputMap.get_shape()[-1]
            _layer1 = self.convBnReluLayer(inputMap, out_channel=in_channel, ksize=ksize, stride=1,
                                           scope_name='_layer1', is_training=is_training, use_bn=False)
            _layer2 = self.convBnReluLayer(_layer1, out_channel=in_channel * 2, ksize=ksize, stride=1,
                                           scope_name='_layer2', is_training=is_training, use_bn=True)
            _layer3 = self.convBnReluLayer(_layer2, out_channel=in_channel * 4, ksize=ksize, stride=1,
                                           scope_name='_layer3', is_training=is_training, use_bn=True)
            _layer4 = self.convBnReluLayer(_layer3, out_channel=in_channel * 8, ksize=ksize, stride=1,
                                           scope_name='_layer4', is_training=is_training, use_bn=True)

            _gap = self.globalPoolLayer(_layer4, scope_name='_gap')
            _flatten = self.flattenLayer(_gap, scope_name='_flatten')

            _fc1 = self.fcLayer(_flatten, out_channel=512, scope_name='_fc1')
            _fc2 = self.fcLayer(_fc1, out_channel=512, scope_name='_fc2')
            _fc3 = self.fcLayer(_fc2, out_channel=1, scope_name='_fc3')

        return _fc3

    def build_model(self):
        self.x_source = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_source')
        self.x_target = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_target')

        self.y_source = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y_source')
        self.y_target = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y_target')

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        tf.summary.image('source_input', self.x_source)
        tf.summary.image('target_input', self.x_target)

        self.source_featureMaps = self.resnet_model(
            inputMap=self.x_source,
            model_name='source_encoder',
            ksize=3,
            unit_num1=3,
            unit_num2=3,
            unit_num3=3,
            out_channel1=16,
            out_channel2=32,
            out_channel3=64,
            reuse=False)

        self.target_featureMaps = self.resnet_model(
            inputMap=self.x_target,
            model_name='target_encoder',
            ksize=3,
            unit_num1=3,
            unit_num2=3,
            unit_num3=3,
            out_channel1=16,
            out_channel2=32,
            out_channel3=64,
            reuse=False)

        self.source_pred, self.source_pred_softmax = self.classifier(self.source_featureMaps, scope_name='classifier',
                                                                     reuse=False)
        self.target_pred, self.target_pred_softmax = self.classifier(self.target_featureMaps, scope_name='classifier',
                                                                     reuse=True)

        self.D_source = self.Discriminator(self.source_featureMaps,
                                           ksize=3,
                                           scope_name='Discriminator',
                                           is_training=self.is_training,
                                           reuse=False)
        self.D_target = self.Discriminator(self.target_featureMaps,
                                           ksize=3,
                                           scope_name='Discriminator',
                                           is_training=self.is_training,
                                           reuse=True)

        with tf.variable_scope('loss'):
            # generator losses
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_target,
                labels=tf.ones_like(self.D_target)))

            # discriminator losses
            self.source_dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_source,
                labels=tf.ones_like(self.D_source)))
            self.target_dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_target,
                labels=tf.zeros_like(self.D_target)))

            self.d_loss = self.source_dloss + self.target_dloss

            tf.summary.scalar('g_loss', self.g_loss)
            tf.summary.scalar('d_loss', self.d_loss)

        with tf.variable_scope('optimization_variables'):
            self.tar_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_encoder')
            self.dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

            # 重载变量域
            self.src_reload_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='source_encoder')
            self.cla_reload_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')
            self.tar_reload_var_pre = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_encoder')
            self.tar_reload_var = {}
            for i in self.src_reload_var:
                for j in self.tar_reload_var_pre:
                    if i.name[i.name.find('/') + 1:] in j.name[j.name.find('/') + 1:]:
                        self.tar_reload_var[i.name[:-2]] = j

        with tf.variable_scope('optimize'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')):
                self.d_train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.d_loss,
                                                                                      var_list=self.dis_var)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='target_encoder')):
                self.g_train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.g_loss,
                                                                                      var_list=self.tar_var)
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
            self.distribution_source = [tf.argmax(self.y_source, 1), tf.argmax(self.source_pred_softmax, 1)]
            self.distribution_target = [tf.argmax(self.y_target, 1), tf.argmax(self.target_pred_softmax, 1)]

            self.correct_prediction_source = tf.equal(self.distribution_source[0], self.distribution_source[1])
            self.correct_prediction_target = tf.equal(self.distribution_target[0], self.distribution_target[1])

            self.accuracy_source = tf.reduce_mean(tf.cast(self.correct_prediction_source, 'float'))
            self.accuracy_target = tf.reduce_mean(tf.cast(self.correct_prediction_target, 'float'))

    def getBatchData(self):
        _src_tr_img_batch, _src_tr_lab_batch = init.next_batch(self.source_training_data[0],
                                                               self.source_training_data[1], self.bs)
        _tar_tr_img_batch = init.next_batch_unpaired(self.target_training_data, self.bs)

        feed_dict = {self.x_source: _src_tr_img_batch,
                     self.y_source: _src_tr_lab_batch,
                     self.x_target: _tar_tr_img_batch,
                     self.is_training: True}
        feed_dict_eval = {self.x_source: _src_tr_img_batch,
                          self.y_source: _src_tr_lab_batch,
                          self.x_target: _tar_tr_img_batch,
                          self.is_training: False}

        return feed_dict, feed_dict_eval

    def train(self):
        print('Initialize parameters')
        self.sess.run(tf.global_variables_initializer())
        print('Global variables initialization finished')

        print('Reload parameters')
        self.src_encoder_reloadSaver = tf.train.Saver(var_list=self.src_reload_var)
        self.tar_encoder_reloadSaver = tf.train.Saver(var_list=self.tar_reload_var)
        self.classifier_reloadSaver = tf.train.Saver(var_list=self.cla_reload_var)

        self.src_encoder_reloadSaver.restore(self.sess, self.reloadPath)
        self.tar_encoder_reloadSaver.restore(self.sess, self.reloadPath)
        self.classifier_reloadSaver.restore(self.sess, self.reloadPath)
        print('source encoder, target encoder and classifier have been successfully reloaded !')

        # 开始训练
        self.itr_epoch = len(self.source_training_data[0]) // self.bs

        source_training_acc = 0.0
        g_loss = 0.0
        d_loss = 0.0

        for e in range(1, self.eps + 1):
            for itr in range(self.itr_epoch):
                feed_dict_train, feed_dict_eval = self.getBatchData()
                _ = self.sess.run(self.d_train_op, feed_dict=feed_dict_train)
                _ = self.sess.run(self.g_train_op, feed_dict=feed_dict_train)

                _training_accuracy, _g_loss, _d_loss = self.sess.run([self.accuracy_source, self.g_loss, self.d_loss],
                                                                     feed_dict=feed_dict_eval)

                source_training_acc += _training_accuracy
                g_loss += _g_loss
                d_loss += _d_loss

            summary = self.sess.run(self.merged, feed_dict=feed_dict_eval)

            source_training_acc = float(source_training_acc / self.itr_epoch)
            g_loss = float(g_loss / self.itr_epoch)
            d_loss = float(d_loss / self.itr_epoch)

            log1 = "Epoch: [%d], Training Accuracy: [%g], G Loss: [%g], D Loss: [%g], Time: [%s]" % (
                e, source_training_acc, g_loss, d_loss, time.ctime(time.time()))

            self.plt_epoch.append(e)
            self.plt_training_accuracy.append(source_training_acc)
            self.plt_g_loss.append(g_loss)
            self.plt_d_loss.append(d_loss)

            utils.plotAccuracy(x=self.plt_epoch,
                               y1=self.plt_training_accuracy,
                               y2=None,
                               figName=self.model,
                               line1Name='training',
                               line2Name='',
                               savePath=self.ckptDir)

            utils.plotLoss(x=self.plt_epoch,
                           y1=self.plt_g_loss,
                           y2=self.plt_d_loss,
                           figName=self.model,
                           line1Name='g loss',
                           line2Name='d loss',
                           savePath=self.ckptDir)

            init.save2file(log1, self.ckptDir, self.model)

            self.writer.add_summary(summary, e)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e))

            eval.test_procedure(self.source_test_data, distribution_op=self.distribution_source,
                                inputX=self.x_source,
                                inputY=self.y_source, mode='source', num_class=self.num_class, batch_size=self.bs,
                                session=self.sess, is_training=self.is_training, ckptDir=self.ckptDir, model=self.model)
            eval.test_procedure(self.target_test_data, distribution_op=self.distribution_target,
                                inputX=self.x_target,
                                inputY=self.y_target, mode='target', num_class=self.num_class, batch_size=self.bs,
                                session=self.sess, is_training=self.is_training, ckptDir=self.ckptDir, model=self.model)

            source_training_acc = 0.0
            g_loss = 0.0
            d_loss = 0.0
