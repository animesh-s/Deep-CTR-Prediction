import os
import argparse
import datetime
import train

parser = argparse.ArgumentParser(description='CTR Predictor')
# learning
parser.add_argument('-modeltype', type=str, default='XGBoost', help='model type')
#parser.add_argument('-lr', type=float, default=0.00859, help='learning rate to use for training')
#parser.add_argument('-ae-lr', type=float, default=0.0001, help='learning rate for autoencoder to use for training')
#parser.add_argument('-weight-decay', type=float, default=0.00019, help='weight decay to use for training')
#parser.add_argument('-max-depth', type=int, default=32, help='max depth to use for training')
#parser.add_argument('-num-rounds', type=int, default=16, help='number of rounds to use for training')
#parser.add_argument('-factor', type=int, default=100, help='factor for feature embeddings')
parser.add_argument('-num-models', type=int, default=50, help='number of models for cross validation [default: 50]')
parser.add_argument('-epochs', type=int, default=5, help='number of epochs for train [default: 5]')
parser.add_argument('-iterations', type=int, default=None, help='number of iterations for train [default: None]')
parser.add_argument('-imbalance-factor', type=int, default=9, help='class imbalance factor for training [default: 9]')
parser.add_argument('-log-interval',  type=int, default=6480,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-plot-interval',  type=int, default=300000,   help='how many steps to wait before plotting training status [default: 1]')
parser.add_argument('-save-interval', type=int, default=300000, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='../Snapshots', help='where to save the snapshots')
parser.add_argument('-plot-dir', type=str, default='../Plots', help='where to save the plots')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
#device
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
args = parser.parse_args()


# update args and print
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args.plot_dir = os.path.join(args.plot_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
args.filepath = os.path.join(args.save_dir, 
                             args.modeltype + '_' + str(args.static) + '.txt')


print("\nParameters:")    
f = open(args.filepath, 'a')
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))
    f.write('%s = %s\n' %(attr.upper(), value))
f.close()


train.cross_validation(args)
