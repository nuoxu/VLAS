"""
test the greedy classification Network
"""
import sys
sys.path.insert(0, '.')
import os
import random
import torch
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import argparse
from torch.autograd import Variable
#from tensorboard_logger import configure, log_value
from torch.distributions import Bernoulli
import csv
from utils import utils
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description='PolicyNetworkTraining')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--data_dir', default='dataset/', help='data directory')
parser.add_argument('--dataset', choices=['DOTA', 'xView'], default='DOTA', help='name of dataset')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='model/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')#256
parser.add_argument('--img_size', type=int, default=448, help='PN Image Size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--num_workers', type=int, default=8, help='Number of Workers')
parser.add_argument('--test_epoch', type=int, default=10, help='At every N epoch test the network')
parser.add_argument('--parallel', action='store_true', default=True, help='use multiple GPUs for training')
parser.add_argument('--alpha', type=float, default=0.6, help='probability bounding factor')
parser.add_argument('--beta', type=float, default=0.1, help='Coarse detector increment')
parser.add_argument('--sigma', type=float, default=0.5, help='cost for patch use')
parser.add_argument('--num_actions', choices=[36, 49, 64, 81, 100], type=int, default=64, help='total number of action/grid')
parser.add_argument('--multiclass', action='store_true', default=False, help='multi-class in language instruction')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)
if not os.path.exists(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)):
    os.makedirs(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions))

def test(epoch, search_budget=12):
    agent.eval()
    targets_found, metrics, policies, set_labels, num_targets, num_search = list(), [], [], [], list(), list()
    
    ant_steps = {}
    dataset_name = args.dataset
    with open('dataset/{}_class_labels.txt'.format(dataset_name.lower())) as f:
        for row in csv.reader(f):
            ant_steps[row[0].split(":")[1]] = []

    for batch_idx, (inputs, targets, classnames, numbers) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs = Variable(inputs, volatile=True)
        if not args.parallel:
            inputs = inputs.cuda()
        
        # Actions by the Policy Network
        probs = F.sigmoid(agent(inputs))

        # Sample the policy from the agents output
        policy = probs.data.clone()
        ### greedy policy
        _, b = torch.topk(probs, search_budget, dim=1)
        decision_tensor = torch.zeros(int(inputs.shape[0]),args.num_actions).cuda()
        for i in range(int(inputs.shape[0])):
            decision_tensor[i,:].index_fill_(0, b[i,:].cuda(), 1)
        policy = decision_tensor
        ### end
        policy = Variable(policy)

        if classnames[0] not in ant_steps.keys():
            ant_steps[classnames[0]] = []
        for policy_sample in torch.nonzero(policy[0]):
            ant_steps[classnames[0]].append(numbers[0][policy_sample[0]])

        targets_, total_targets, total_search = utils.compute_reward_test(targets, policy.data, args.beta, args.sigma)
        
        for sample_id in range(int(inputs.shape[0])):
            temp = int(torch.sum(targets[sample_id,:]))
            if (temp > search_budget):
                num_targets.append(search_budget)
            else:
                num_targets.append(temp)

        targets_found.append(targets_)

    
    recall = sum(targets_found) / sum(num_targets)

    final_ant = []
    final_mant = []
    for k in ant_steps.keys():
        final_ant.append(torch.mean(torch.stack(ant_steps[k])))
        final_mant += ant_steps[k]
    final_mant = torch.mean(torch.stack(final_mant))
    
    # store the log in different log file
    with open(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)+'/'+'TEST_GC_log.txt','a') as f:
        f.write('Test - Recall: %.2f | mANT: %.2f | SB: %.2F \n' % (recall, final_mant, search_budget))
        for idx, k in enumerate(ant_steps.keys()):
            f.write('       ANT - %.2f : %s \n' % (final_ant[idx], k))
    
    print('Test - Recall: %.2f | mANT: %.2F | SB: %.2f' % (recall, final_mant, search_budget))

#--------------------------------------------------------------------------------------------------------#
testset = utils.get_datasetVAS_test(args.img_size, args.data_dir+'/'+args.dataset+'/', args.num_actions, args.multiclass)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

agent = utils.get_model(args.num_actions)

# ---- Load the pre-trained model ----------------------
checkpoint = torch.load(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)+"/GC.pt")
agent.load_state_dict(checkpoint['agent'])  
start_epoch = checkpoint['epoch'] + 1
print('loaded agent from %s' % args.load)

start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print('loaded agent from %s' % args.load)

# Parallelize the models if multiple GPUs available - Important for Large Batch Size to Reduce Variance
if args.parallel:
    agent = nn.DataParallel(agent)
agent.cuda()

# Update the parameters of the policy network
optimizer = optim.Adam(agent.parameters(), lr=args.lr)

# Start testing
budgets = [int(args.num_actions/6.), int(args.num_actions/3.), int(args.num_actions/2.)]
for budget in budgets:
    test(0, budget)