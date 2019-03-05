import sys

sys.path.append('../Data_Initialization/')
import tensorflow as tf
import tensorflow.contrib.layers as layers
import Initialization as init
import evaluation_function as eval
import time


class cpb_dastage3_model(object):
    def __init__(self, model_name, sess, train_data, val_data, tst_data, epoch, restore_epoch, reloadPath, num_class,
                 learning_rate, lambda_para, batch_size, img_height, img_width):

        self.sess = sess
        self.source_training_data = train_data[0]
        self.source_validation_data = val_data
        self.source_test_data = tst_data[0]
        self.target_training_data = train_data[1]
        self.target_test_data = tst_data[1]
        self.eps = epoch
        self.res_eps = restore_epoch
        self.reloadPath = reloadPath
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.lr = learning_rate
        self.lbd = lambda_para
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
        init.save2file('learning rate : %g' % self.lr, self.ckptDir, self.model)
        init.save2file('lambda : %g' % self.lbd, self.ckptDir, self.model)
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

    def deconvLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_result = tf.layers.conv2d_transpose(inputMap, filters=out_channel, kernel_size=(ksize, ksize),
                                                     strides=(stride, stride), padding=padding, use_bias=False,
                                                     name='conv_result',
                                                     kernel_initializer=layers.variance_scaling_initializer())

            tf.summary.histogram('deconv_result', conv_result)

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

    def dropoutLayer(self, inputMap, scope_name, keep_rate):
        with tf.variable_scope(scope_name):
            return tf.nn.dropout(inputMap, keep_prob=keep_rate)

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

    def convBnReluLayer(self, inputMap, ksize, stride, out_channel, scope_name, is_training, bn, relu):
        if relu:
            activation = self.reluLayer
        else:
            activation = self.lreluLayer

        with tf.variable_scope(scope_name):
            _conv = self.convLayer(inputMap, out_channel=out_channel, ksize=ksize, stride=stride,
                                   scope_name='_conv')
            if bn:
                _conv = self.bnLayer(_conv, scope_name='_bn', is_training=is_training)
            _relu = activation(_conv, scope_name='_relu')

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

    def resnet_model(self, input_x, model_name, ksize, unit_num1, unit_num2, unit_num3, out_channel1, out_channel2,
                     out_channel3, reuse):
        with tf.variable_scope(model_name, reuse=reuse):
            _conv = self.convLayer(input_x, out_channel1, ksize=ksize, stride=1, scope_name='stage0_conv')
            _bn = self.bnLayer(_conv, scope_name='stage0_bn', is_training=self.is_training)
            _relu = self.reluLayer(_bn, scope_name='stage0_relu')

            stage1_out = self.residualStageLayer(inputMap=_relu,
                                                 ksize=ksize,
                                                 out_channel=out_channel1,
                                                 unit_num=unit_num1,
                                                 section_name='stage1',
                                                 down_sampling=False,
                                                 first_conv=True,
                                                 is_training=self.is_training)

            stage2_out = self.residualStageLayer(inputMap=stage1_out,
                                                 ksize=ksize,
                                                 out_channel=out_channel2,
                                                 unit_num=unit_num2,
                                                 section_name='stage2',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            stage3_out = self.residualStageLayer(inputMap=stage2_out,
                                                 ksize=ksize,
                                                 out_channel=out_channel3,
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

            return y_pred, y_pred_softmax, [_relu, stage1_out, stage2_out, stage3_out]

    def ContentPreservedBlock(self, inputMapZoo, ksize, scope_name, is_training, adjustScale, reuse):
        with tf.variable_scope(scope_name, reuse=reuse):
            input_shape = inputMapZoo[1].get_shape().as_list()

            conv1 = self.convBnReluLayer(inputMapZoo[1],
                                         ksize=ksize,
                                         stride=1,
                                         out_channel=input_shape[-1],
                                         scope_name='conv',
                                         is_training=is_training,
                                         bn=True,
                                         relu=True)

            residual_section = self.residualStageLayer(conv1,
                                                       ksize=ksize,
                                                       out_channel=input_shape[-1],
                                                       unit_num=3,
                                                       section_name='residual_stage',
                                                       down_sampling=False,
                                                       first_conv=True,
                                                       is_training=is_training)

            if adjustScale:
                conv2 = self.deconvLayer(inputMap=residual_section,
                                         out_channel=input_shape[-1] // 2,
                                         ksize=3,
                                         stride=2,
                                         scope_name='deconv')
            else:
                conv2 = self.convLayer(inputMap=residual_section,
                                       out_channel=input_shape[-1],
                                       ksize=3,
                                       stride=1,
                                       scope_name='conv')
            return conv2

    def Discriminator(self, inputMap, ksize, scope_name, is_training, keep_rate, reuse):
        with tf.variable_scope(scope_name, reuse=reuse):
            in_channel = inputMap.get_shape()[-1]
            _layer1 = self.convBnReluLayer(inputMap, out_channel=in_channel, ksize=ksize, stride=1,
                                           scope_name='_layer1', is_training=is_training, bn=False, relu=False)
            _layer2 = self.convBnReluLayer(_layer1, out_channel=in_channel * 2, ksize=ksize, stride=1,
                                           scope_name='_layer2', is_training=is_training, bn=True, relu=False)
            _layer3 = self.convBnReluLayer(_layer2, out_channel=in_channel * 4, ksize=ksize, stride=1,
                                           scope_name='_layer3', is_training=is_training, bn=True, relu=False)
            _layer4 = self.convBnReluLayer(_layer3, out_channel=in_channel * 8, ksize=ksize, stride=1,
                                           scope_name='_layer4', is_training=is_training, bn=True, relu=False)
            _fc1 = self.fcLayer(_layer4, out_channel=512, scope_name='_fc1')
            _dp1 = self.dropoutLayer(_fc1, scope_name='_dp1', keep_rate=keep_rate)
            _fc2 = self.fcLayer(_dp1, out_channel=512, scope_name='_fc2')
            _dp2 = self.dropoutLayer(_fc2, scope_name='_dp2', keep_rate=keep_rate)
            _fc3 = self.fcLayer(_dp2, out_channel=1, scope_name='_fc3')

        return _fc3

    def build_model(self):
        self.x_source = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_source')
        self.x_target = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_target')

        self.y_source = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y_source')
        self.y_target = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y_target')

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_rate = tf.placeholder(tf.float32, name='keep_rate')

        tf.summary.image('source_input', self.x_source)
        tf.summary.image('target_input', self.x_target)

        self.pred_source, self.pred_softmax_source, self.feature_source = self.resnet_model(
            input_x=self.x_source,
            model_name='classification_model',
            ksize=3,
            unit_num1=3,
            unit_num2=3,
            unit_num3=3,
            out_channel1=16,
            out_channel2=32,
            out_channel3=64,
            reuse=False)

        self.pred_target, self.pred_softmax_target, self.feature_target = self.resnet_model(
            input_x=self.x_target,
            model_name='classification_model',
            ksize=3,
            unit_num1=3,
            unit_num2=3,
            unit_num3=3,
            out_channel1=16,
            out_channel2=32,
            out_channel3=64,
            reuse=True)

        # feature consistency in scale 3
        self.tar_fea_cons_sc3 = self.ContentPreservedBlock([self.feature_target[2], self.feature_target[3]],
                                                           ksize=3,
                                                           scope_name='cpb3',
                                                           is_training=self.is_training,
                                                           adjustScale=True,
                                                           reuse=False)

        self.D_src_fea_sc3 = self.Discriminator(self.feature_source[3],
                                                ksize=3,
                                                scope_name='d3',
                                                is_training=self.is_training,
                                                keep_rate=self.keep_rate,
                                                reuse=False)

        self.D_tar_fea_sc3 = self.Discriminator(self.feature_target[3],
                                                ksize=3,
                                                scope_name='d3',
                                                is_training=self.is_training,
                                                keep_rate=self.keep_rate,
                                                reuse=True)

        with tf.variable_scope('loss'):
            # supervised loss
            self.supervised_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.pred_source, labels=self.y_source))

            # feature consistency loss
            self.cpb_input = self.tar_fea_cons_sc3
            self.cpb_output = self.feature_target[2]
            self.cpb_loss = tf.losses.absolute_difference(labels=self.cpb_output, predictions=self.cpb_input)

            # g losses
            self.target_gloss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_tar_fea_sc3,
                                                        labels=tf.ones_like(self.D_tar_fea_sc3)))

            self.g_loss = self.target_gloss + self.cpb_loss + self.lbd * self.supervised_loss

            # d losses
            self.source_dloss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_src_fea_sc3,
                                                        labels=tf.ones_like(self.D_src_fea_sc3)))
            self.target_dloss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_tar_fea_sc3,
                                                        labels=tf.zeros_like(self.D_tar_fea_sc3)))

            self.d_loss = self.source_dloss + self.target_dloss

            tf.summary.scalar('Loss/supervised_loss', self.supervised_loss)
            tf.summary.scalar('Loss/target_gloss', self.target_gloss)
            tf.summary.scalar('Loss/total_gloss', self.g_loss)
            tf.summary.scalar('Loss/source_dloss', self.source_dloss)
            tf.summary.scalar('Loss/target_dloss', self.target_dloss)
            tf.summary.scalar('Loss/total_dloss', self.d_loss)
            tf.summary.scalar('Loss/cpb_loss', self.cpb_loss)

        with tf.variable_scope('optimization_variables'):
            self.t_var = [var for var in tf.trainable_variables()]

            self.cpb_var = [var for var in self.t_var if 'cpb3' in var.name]
            self.d_var = [var for var in self.t_var if 'd3' in var.name]

            self.encoder_var = [var for var in self.t_var if 'classification_model' in var.name]

        with tf.variable_scope('reload_variables'):
            self.g_var = [var for var in tf.global_variables()]
            self.g_var_reload = [var for var in self.g_var if 'classification_model' in var.name]

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.cpb_trainOp = tf.train.AdamOptimizer(self.lr).minimize(self.cpb_loss,
                                                                            var_list=self.cpb_var)
                self.d_trainOp = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.d_loss,
                                                                                     var_list=self.d_var)
                self.g_trainOp = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.g_loss,
                                                                                     var_list=self.encoder_var)
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
            self.distribution_source = [tf.argmax(self.y_source, 1), tf.argmax(self.pred_softmax_source, 1)]
            self.distribution_target = [tf.argmax(self.y_target, 1), tf.argmax(self.pred_softmax_target, 1)]

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
                     self.is_training: True,
                     self.keep_rate: 0.5}
        feed_dict_eval = {self.x_source: _src_tr_img_batch,
                          self.y_source: _src_tr_lab_batch,
                          self.x_target: _tar_tr_img_batch,
                          self.is_training: False,
                          self.keep_rate: 0.5}

        return feed_dict, feed_dict_eval

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.reloadSaver = tf.train.Saver(var_list=self.g_var_reload)
        self.reloadSaver.restore(self.sess, self.reloadPath)
        print('Pre-trained classification model has been successfully reloaded !')

        self.itr_epoch = len(self.source_training_data[0]) // self.bs
        self.total_iteration = self.eps * self.itr_epoch

        source_training_acc = 0.0
        cpb_loss = 0.0
        g_loss = 0.0
        d_loss = 0.0

        for itr in range(1, self.total_iteration + 1):
            feed_dict_train, feed_dict_eval = self.getBatchData()
            _, _ = self.sess.run([self.cpb_trainOp, self.d_trainOp], feed_dict=feed_dict_train)

            feed_dict_train, feed_dict_eval = self.getBatchData()
            _ = self.sess.run(self.g_trainOp, feed_dict=feed_dict_train)

            _training_accuracy, _cpb_loss, _g_loss, _d_loss = self.sess.run(
                [self.accuracy_source, self.cpb_loss, self.g_loss, self.d_loss], feed_dict=feed_dict_eval)

            source_training_acc += _training_accuracy
            cpb_loss += _cpb_loss
            g_loss += _g_loss
            d_loss += _d_loss

            if itr % self.itr_epoch == 0:
                _current_eps = int(itr / self.itr_epoch)
                summary = self.sess.run(self.merged, feed_dict=feed_dict_eval)

                source_training_acc = float(source_training_acc / self.itr_epoch)
                cpb_loss = float(cpb_loss / self.itr_epoch)
                g_loss = float(g_loss / self.itr_epoch)
                d_loss = float(d_loss / self.itr_epoch)

                log = "Epoch: [%d], Training Accuracy: [%.4f], G Loss: [%.4f], D Loss: [%.4f], " \
                      "CPB Loss: [%.4f], Time: [%s]" % (
                          _current_eps, source_training_acc, g_loss, d_loss, cpb_loss, time.ctime(time.time()))

                init.save2file(log, self.ckptDir, self.model)

                self.writer.add_summary(summary, _current_eps)

                self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(_current_eps))

                eval.test_procedure_DA(self.source_test_data, distribution_op=self.distribution_source,
                                       inputX=self.x_source, inputY=self.y_source, mode='source',
                                       num_class=self.num_class, batch_size=self.bs, session=self.sess,
                                       is_training=self.is_training, ckptDir=self.ckptDir, model=self.model,
                                       keep_rate=self.keep_rate)
                eval.test_procedure_DA(self.target_test_data, distribution_op=self.distribution_target,
                                       inputX=self.x_target, inputY=self.y_target, mode='target',
                                       num_class=self.num_class, batch_size=self.bs, session=self.sess,
                                       is_training=self.is_training, ckptDir=self.ckptDir, model=self.model,
                                       keep_rate=self.keep_rate)

                source_training_acc = 0.0
                cpb_loss = 0.0
                g_loss = 0.0
                d_loss = 0.0
