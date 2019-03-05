import sys

sys.path.append('../Data_Initialization/')
import os
from ADDA_model_step1 import DA_Model_step1
from ADDA_model_step2 import DA_Model_step2
import Initialization as init
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-model_name', required=True, help='[the name of the model]')
parser.add_argument('-train_phase', required=True, help='[whether to train or test the model]')
parser.add_argument('-gpu', required=True, help='[set particular gpu for calculation]')
parser.add_argument('-data_domain', required=True, help='[choose the data domain between source and target]')
parser.add_argument('-step', type=int, required=True, help='[set the step of ADDA algorithm]')

parser.add_argument('-epoch', default=200, type=int)
parser.add_argument('-num_class', default=6, type=int)
parser.add_argument('-learning_rate', default=2e-4, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-img_height', default=32, type=int)
parser.add_argument('-img_width', default=32, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

src_data, tar_data = init.loadData(data_domain=args.data_domain)
src_training, src_validation, src_test = src_data
tar_training, tar_test = tar_data

if args.data_domain == 'Source':
    reloadPath = '../checkpoint/adda_s2t_step1/adda_s2t_step1-85'
elif args.data_domain == 'Target':
    reloadPath = '../checkpoint/adda_t2s_step1/adda_t2s_step1-9'
else:
    reloadPath = ''

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if args.step == 1:
    with tf.Session(config=config) as sess:
        res_model = DA_Model_step1(model_name=args.model_name,
                                   sess=sess,
                                   train_data=[src_training, tar_training],
                                   val_data=src_validation,
                                   tst_data=[src_test, tar_test],
                                   epoch=args.epoch,
                                   num_class=args.num_class,
                                   learning_rate=args.learning_rate,
                                   batch_size=args.batch_size,
                                   img_height=args.img_height,
                                   img_width=args.img_width,
                                   train_phase=args.train_phase,
                                   step=args.step)

        if args.train_phase == 'Train':
            res_model.train()

        if args.train_phase == 'Test':
            res_model.test(reloadPath)

if args.step == 2:
    with tf.Session(config=config) as sess:
        res_model = DA_Model_step2(model_name=args.model_name,
                                   sess=sess,
                                   train_data=[src_training, tar_training],
                                   val_data=[src_validation],
                                   tst_data=[src_test, tar_test],
                                   epoch=args.epoch,
                                   reloadPath=reloadPath,
                                   num_class=args.num_class,
                                   learning_rate=args.learning_rate,
                                   batch_size=args.batch_size,
                                   img_height=args.img_height,
                                   img_width=args.img_width,
                                   train_phase=args.train_phase,
                                   step=args.step)

        if args.train_phase == 'Train':
            res_model.train()
