import yaml
import torch
import argparse

from models import DeepONet_darcy, Improved_DeepOnet_darcy, DCON_darcy
from data import generate_darcy_dataloader
from darcy_utils import train

# define arguements
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--data', type=str, default='Darcy_star')
parser.add_argument('--model', type=str, default='DON')
args = parser.parse_args()
print('Model forward phase: {}'.format(args.phase))
print('Using dataset: {}'.format(args.data))
print('Using model: {}'.format(args.model))

# extract configuration
with open(r'./configs/{}_{}.yaml'.format(args.model, args.data), 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

# load the data 
coors, BC_flags, num_bc_nodes, train_loader, val_loader, test_loader = generate_darcy_dataloader(config)

# define model
if args.model == 'DCON':
    model = DCON_darcy(config)
if args.model == 'DON':
    model = DeepONet_darcy(config, num_bc_nodes)
if args.model == 'IDON':
    model = Improved_DeepOnet_darcy(config, num_bc_nodes)

# define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# impletment model training
train(args, config, model, device, (train_loader, val_loader, test_loader), coors, BC_flags)


