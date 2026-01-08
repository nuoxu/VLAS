#### VAS TEST
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
from torch.distributions.categorical import Categorical
import csv
from utils import utils

parser = argparse.ArgumentParser(description='PolicyNetworkTraining')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') 
parser.add_argument('--data_dir', default='dataset/', help='data directory')
parser.add_argument('--dataset', choices=['DOTA', 'xView'], default='DOTA', help='name of dataset')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='model/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--img_size', type=int, default=448, help='PN Image Size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--num_workers', type=int, default=8, help='Number of Workers')
parser.add_argument('--test_epoch', type=int, default=10, help='At every N epoch test the network')
parser.add_argument('--parallel', action='store_true', default=True, help='use multiple GPUs for training')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--beta', type=float, default=0.1, help='Coarse detector increment')
parser.add_argument('--sigma', type=float, default=0.5, help='cost for patch use')
parser.add_argument('--num_actions', choices=[36, 49, 64, 81, 100], type=int, default=64, help='total number of action/grid')
parser.add_argument('--multiclass', action='store_true', default=False, help='multi-class in language instruction')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)
if not os.path.exists(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)):
    os.makedirs(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions))

# the value "6" comes from 6 * 6 grid size, in case of 8 * 8 grid structure, 6 need to be replaced with 8
def coord(x):
    x_= x//6 + 1   #7, 5, 9  8
    y_= x%6        #7, 6, 9  6
    return (x_,y_)

# test the agent's performance on VAS setting    
def test(epoch, search_budget=12): 
    # set the agent in evaluation mode
    agent.eval()
    # initialize lists to store search outcomes
    targets_found, num_targets = list(), list()
    num_image = 0
    ant_steps = {}
    dataset_name = args.dataset
    with open('dataset/{}_class_labels.txt'.format(dataset_name.lower())) as f:
        for row in csv.reader(f):
            ant_steps[row[0].split(":")[1]] = []
    # iterate over the test data
    for batch_idx, (inputs, targets, classnames, numbers) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        num_image += 1
        inputs = Variable(inputs, volatile=True)
        if not args.parallel:
            inputs = inputs.cuda()
        
        # stores the information of previous search queries
        search_info = torch.zeros((int(inputs.shape[0]), args.num_actions)).cuda()
        # stores the information of previously selected grids as target 
        mask_info = torch.ones((int(inputs.shape[0]), args.num_actions)).cuda()
        #store the information about the remaining query
        query_info = torch.zeros(int(inputs.shape[0])).cuda()

        # Start an episode
        policy_loss = []; search_history = []; reward_history = []
        travel_cost = remain_cost = 75
        for step_ in range(search_budget): 
            #while remain_cost > 0:
            query_remain = search_budget - step_
            # number of query left
            query_left = torch.add(query_info, query_remain).cuda()
            # action taken by agent
            logit = agent.forward(inputs, search_info)
            # get the probability distribution over grids
            probs = F.softmax(logit, dim=1)
            # assign 0 probability to those grids that is already queried by agent
            mask_probs = probs * mask_info.clone()  
            # Sample the grid that corresponds to highest probability of being target
            policy_sample = torch.argmax(mask_probs, dim=1) 
            if classnames[0] not in ant_steps.keys():
                ant_steps[classnames[0]] = []
            ant_steps[classnames[0]].append(numbers[0][policy_sample[0]])
            ################# travel
            # compute the reward for the agent's action
            reward_update = utils.compute_reward(targets, policy_sample.data, args.beta, args.sigma)
            # get the outcome of an action in order to compute ESR/SR 
            reward_sample = utils.compute_reward_batch(targets, policy_sample.data, args.beta, args.sigma)    
            # Update search info and mask info after every query
            for sample_id in range(int(inputs.shape[0])):
                # update the search info based on the reward
                search_info[sample_id, int(policy_sample[sample_id].data)] = int(reward_update[sample_id])     
                # update the mask info based on the current action
                mask_info[sample_id, int(policy_sample[sample_id].data)] = 0
            # store the episodic reward in the list
            reward_history.append(reward_sample)
            
        # obtain the number of target grid in each sample (maximum value is search budget)
        for target_id in range(int(targets.shape[0])):
            temp = int(targets[target_id,:].sum())
            if temp > search_budget:
                num_targets.append(search_budget)
            else:
                num_targets.append(temp)
        # concat the episodic reward over the samples in a batch
        batch_reward = torch.cat(reward_history).sum() 
        targets_found.append(batch_reward)    
    ## compute ESR/ANT
    temp_recall = torch.sum(torch.stack(targets_found))
    recall = temp_recall / sum(num_targets)  
    sr_budget = temp_recall / num_image
    print ("travel sr:", sr_budget)
    final_ant = []
    final_mant = []
    for k in ant_steps.keys():
        final_ant.append(torch.mean(torch.stack(ant_steps[k])))
        final_mant += ant_steps[k]
    final_mant = torch.mean(torch.stack(final_mant))
    # store the log in different log file
    with open(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)+'/'+'TEST_VAS_log.txt','a') as f:
        f.write('Test - Recall: %.2f | mANT: %.2f | SB: %.2F \n' % (recall, final_mant, search_budget))
        for idx, k in enumerate(ant_steps.keys()):
            f.write('       ANT - %.2f : %s \n' % (final_ant[idx], k))

    print('Test - Recall: %.2F | mANT: %.2F | SB: %.2F' % (recall, final_mant, search_budget))

#--------------------------------------------------------------------------------------------------------#
testset = utils.get_datasetVAS_test(args.img_size, args.data_dir+'/'+args.dataset+'/', args.num_actions, args.multiclass)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# initialize the agent
agent = utils.Model_search_Arch(num_grids=args.num_actions)
# ---- Load the pre-trained model ----------------------
#uncomment this if we want to start training from a pretrained model
#provide the model path
checkpoint = torch.load(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)+"/VAS.pt")
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

# Start training and testing
budgets = [int(args.num_actions/6.), int(args.num_actions/3.), int(args.num_actions/2.)]
for budget in budgets:
    test(0, budget)
