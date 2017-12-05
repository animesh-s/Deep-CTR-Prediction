import os
import argparse
import datetime
import train

parser = argparse.ArgumentParser(description='CTR Predictor')
# learning
parser.add_argument('-lr', type=str, default='0.4, 0.5, 0.6', help='comma-separated learning rates to use for training')
parser.add_argument('-ae-lr', type=str, default='0.0001, 0.001, 0.01', help='comma-separated learning rates for autoencoder to use for training')
parser.add_argument('-weight-decay', type=str, default='0.00001, 0.0001, 0.001', help='comma-separated learning rates for autoencoder to use for training')
parser.add_argument('-max-depth', type=str, default='32, 64, 128', help='comma-separated max depth to use for training')
parser.add_argument('-num-rounds', type=str, default='16, 32, 64', help='comma-separated number of rounds to use for training')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 256]')
parser.add_argument('-log-interval',  type=int, default=30000,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-plot-interval',  type=int, default=500,   help='how many steps to wait before plotting training status [default: 1]')
parser.add_argument('-save-interval', type=int, default=30000, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='../Snapshots', help='where to save the snapshots')
parser.add_argument('-plot-dir', type=str, default='../Plots', help='where to save the plots')
parser.add_argument('-factors', type=str, default='100', help='factor for feature embeddings')
parser.add_argument('-imbalance-factor', type=int, default=1, help='class imbalance factor for training')
# model
parser.add_argument('-static', action='store_true', default=True, help='fix the embedding')
# device
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')

args = parser.parse_args()


# update args and print
args.lr = [float(k) for k in args.lr.split(',')]
args.ae_lr = [float(k) for k in args.ae_lr.split(',')]
args.weight_decay = [float(k) for k in args.weight_decay.split(',')]
args.max_depth = [int(k) for k in args.max_depth.split(',')]
args.num_rounds = [int(k) for k in args.num_rounds.split(',')]
args.factors = [int(k) for k in args.factors.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args.plot_dir = os.path.join(args.plot_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

train.cross_validation(args)