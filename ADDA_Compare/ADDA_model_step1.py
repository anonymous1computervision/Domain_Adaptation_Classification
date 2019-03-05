import sys

sys.path.append('../Data_Initialization/')
import tensorflow as tf
import tensorflow.contrib.layers as layers
import Initialization as init
import ADDA_utils as utils
import evaluation_function as eval
import time


class DA_Model_step1(object):
    def __init__(self, model_name, sess, train_data, val_data, tst_data, epoch, num_class, learning_rate, batch_size,
                 img_height, img_width, train_phase, step):

        self.sess = sess
        self.source_training_data = train_data[0]
        self.target_training_data = train_data[1]
        self.source_validation_data = val_data
        self.source_test_data = tst_data[0]
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
        self.plt_epoch = []
        self.plt_training_accuracy = []
        self.plt_validation_accuracy = []
        self.plt_training_loss = []
        self.plt_validation_loss = []

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

    def classifier(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            _fm_bn = self.bnLayer(inputMap, scope_name='_fm_bn', is_training=self.is_training)
            _fm_relu = self.reluLayer(_fm_bn, scope_name='_fm_relu')
            _fm_pool = self.globalPoolLayer(_fm_relu, scope_name='_fm_gap')
            _fm_flatten = self.flattenLayer(_fm_pool, scope_name='_fm_flatten')

            y_pred = self.fcLayer(_fm_flatten, self.num_class, scope_name='fc_pred')
            y_pred_softmax = tf.nn.softmax(y_pred)

            return y_pred, y_pred_softmax

    def build_model(self):
        self.x_source = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_source')
        tf.summary.image('source_input', self.x_source)
        self.y_source = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y_source')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

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

        self.y_pred, self.y_pred_softmax = self.classifier(self.source_featureMaps, scope_name='classifier')

        with tf.variable_scope('loss'):
            # supervised loss
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y_source))
            tf.summary.scalar('supervised_loss', self.loss)

        with tf.variable_scope('optimization_variables'):
            self.t_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=self.t_var)

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
            self.distribution_source = [tf.argmax(self.y_source, 1), tf.argmax(self.y_pred_softmax, 1)]
            self.correct_prediction_source = tf.equal(self.distribution_source[0], self.distribution_source[1])
            self.accuracy_source = tf.reduce_mean(tf.cast(self.correct_prediction_source, 'float'))

    def getBatchData(self):
        _src_tr_img_batch, _src_tr_lab_batch = init.next_batch(self.source_training_data[0],
                                                               self.source_training_data[1], self.bs)

        feed_dict = {self.x_source: _src_tr_img_batch,
                     self.y_source: _src_tr_lab_batch,
                     self.is_training: True}
        feed_dict_eval = {self.x_source: _src_tr_img_batch,
                          self.y_source: _src_tr_lab_batch,
                          self.is_training: False}

        return feed_dict, feed_dict_eval

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.itr_epoch = len(self.source_training_data[0]) // self.bs

        source_training_acc = 0.0
        source_training_loss = 0.0

        for e in range(1, self.eps + 1):
            for itr in range(self.itr_epoch):
                feed_dict_train, feed_dict_eval = self.getBatchData()
                _ = self.sess.run(self.train_op, feed_dict=feed_dict_train)

                _training_accuracy, _training_loss = self.sess.run([self.accuracy_source, self.loss],
                                                                   feed_dict=feed_dict_eval)

                source_training_acc += _training_accuracy
                source_training_loss += _training_loss

            summary = self.sess.run(self.merged, feed_dict=feed_dict_eval)

            source_training_acc = float(source_training_acc / self.itr_epoch)
            source_training_loss = float(source_training_loss / self.itr_epoch)

            source_validation_acc, source_validation_loss = eval.validation_procedure(
                validation_data=self.source_validation_data, distribution_op=self.distribution_source,
                loss_op=self.loss, inputX=self.x_source, inputY=self.y_source, num_class=self.num_class,
                batch_size=self.bs, is_training=self.is_training, session=self.sess)

            log1 = "Epoch: [%d], Training Accuracy: [%g], Validation Accuracy: [%g], Training Loss: [%g], " \
                   "Validation Loss: [%g], Time: [%s]" % (
                e, source_training_acc, source_validation_acc, source_training_loss, source_validation_loss,
                time.ctime(time.time()))

            self.plt_epoch.append(e)
            self.plt_training_accuracy.append(source_training_acc)
            self.plt_training_loss.append(source_training_loss)
            self.plt_validation_accuracy.append(source_validation_acc)
            self.plt_validation_loss.append(source_validation_loss)

            utils.plotAccuracy(x=self.plt_epoch,
                               y1=self.plt_training_accuracy,
                               y2=self.plt_validation_accuracy,
                               figName=self.model,
                               line1Name='training',
                               line2Name='validation',
                               savePath=self.ckptDir)

            utils.plotLoss(x=self.plt_epoch,
                           y1=self.plt_training_loss,
                           y2=self.plt_validation_loss,
                           figName=self.model,
                           line1Name='training',
                           line2Name='validation',
                           savePath=self.ckptDir)

            init.save2file(log1, self.ckptDir, self.model)

            self.writer.add_summary(summary, e)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e))

            eval.test_procedure(self.source_test_data, distribution_op=self.distribution_source,
                                inputX=self.x_source, inputY=self.y_source, mode='source', batch_size=self.bs,
                                session=self.sess, is_training=self.is_training, ckptDir=self.ckptDir, model=self.model,
                                num_class=self.num_class)

            source_training_acc = 0.0
            source_training_loss = 0.0

    def test(self, reload_path):
        self.saver.restore(self.sess, reload_path)
        eval.test_procedure(self.target_test_data, distribution_op=self.distribution_source,
                            inputX=self.x_source, inputY=self.y_source, mode='source', batch_size=self.bs,
                            session=self.sess, is_training=self.is_training, ckptDir=self.ckptDir, model=self.model,
                            num_class=self.num_class)
