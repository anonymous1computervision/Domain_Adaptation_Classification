import sys

sys.path.append('../Data_Initialization/')
import os
from cyclegan_model import CycleGAN_Model
import Initialization as init
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-model_name', required=True, help='[the name of the model]')
parser.add_argument('-train_phase', required=True, help='[whether to train or test the model]')
parser.add_argument('-gpu', required=True, help='[set particular gpu for calculation]')
parser.add_argument('-data_domain', required=True, help='[choose the data domain between source and target]')
parser.add_argument('-step', type=int, required=True, help='[choose the step of cycle-GAN procedure]')

parser.add_argument('-epoch', default=200, type=int)
parser.add_argument('-num_class', default=6, type=int)
parser.add_argument('-learning_rate', default=2e-4, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-img_height', default=32, type=int)
parser.add_argument('-img_width', default=32, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.data_domain == 'Source':
    reload_path = '../checkpoint/cyclegan_s2t_step1/cyclegan_s2t_step1-88'
elif args.data_domain == 'Target':
    reload_path = '../checkpoint/cyclegan_s2t_step1/cyclegan_s2t_step1-88'
else:
    reload_path = ''

src_data, tar_data = init.loadData(data_domain=args.data_domain)
src_training, src_validation, src_test = src_data
tar_training, tar_test = tar_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    res_model = CycleGAN_Model(model_name=args.model_name,
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
                               step=args.step,
                               data_domain=args.data_domain)

    if args.train_phase == 'Train':
        if args.step == 1:
            res_model.train_step1()
        if args.step == 2:
            res_model.train_step2(reload_path=reload_path)
