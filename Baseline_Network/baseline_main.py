import sys

sys.path.append('../Data_Initialization/')
import os
from baseline_model import ResNet
import Initialization as init
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-model_name', required=True, help='[the name of the model]')
parser.add_argument('-gpu', required=True, help='[set particular gpu for calculation]')
parser.add_argument('-data_domain', required=True, help='[choose the data domain between source and target]')

parser.add_argument('-epoch', default=200, type=int)
parser.add_argument('-restore_epoch', default=0, type=int)
parser.add_argument('-num_class', default=6, type=int)
parser.add_argument('-ksize', default=3, type=int)
parser.add_argument('-out_channel1', default=16, type=int)
parser.add_argument('-out_channel2', default=32, type=int)
parser.add_argument('-out_channel3', default=64, type=int)
parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-img_height', default=32, type=int)
parser.add_argument('-img_width', default=32, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

src_data, tar_data = init.loadData(data_domain=args.data_domain)
src_training, src_validation, src_test = src_data
tar_training, tar_test = tar_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    res_model = ResNet(model_name=args.model_name,
                       sess=sess,
                       train_data=[src_training, tar_training],
                       val_data=src_validation,
                       tst_data=[src_test, tar_test],
                       epoch=args.epoch,
                       restore_epoch=args.restore_epoch,
                       num_class=args.num_class,
                       ksize=args.ksize,
                       out_channel1=args.out_channel1,
                       out_channel2=args.out_channel2,
                       out_channel3=args.out_channel3,
                       learning_rate=args.learning_rate,
                       batch_size=args.batch_size,
                       img_height=args.img_height,
                       img_width=args.img_width)

    res_model.train()
