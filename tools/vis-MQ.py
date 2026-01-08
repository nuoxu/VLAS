
"""
Test MQ-VAS Policy
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
from torch.distributions.categorical import Categorical
from copy import deepcopy as c
import csv
from utils import utils
import pdb
np.set_printoptions(suppress=True, precision=16)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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

def coord(x):
    x_= x//6 + 1   #7, 5, 9  8
    y_= x%6        #7, 6, 9  6
    return (x_,y_)    
    
# test the agent's performance on VAS setting    
def test(epoch, search_budget=12): 
    num_image = 0
    batch_query = 3  
    # set the agent in evaluation mode
    agent.eval()
    # initialize lists to store search outcomes
    dataset_name = args.dataset
    with open('dataset/{}_class_labels.txt'.format(dataset_name.lower())) as f:
        for row in csv.reader(f):
            ant_steps[row[0].split(":")[1]] = []
    # iterate over the test data    
    for batch_idx, (inputs, targets, classnames, numbers) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        num_image += 1
        last_visited_grid = {0: [], 1: [], 2: []}
        ## create a copy of the pretrained agent
        search_agent = c(agent)
        optimizer = optim.Adam(search_agent.parameters(), lr=0.0002) #1 foe 12,15 2 for 18,15 0.0002 0.00001
        inputs = Variable(inputs, volatile=True)                      # batch 12 or 25 cost 0.00001, else 0.0002
        if not args.parallel:
            inputs = inputs.cuda()
        # stores the information of previous search queries
        search_info = torch.zeros((int(inputs.shape[0]), args.num_actions)).cuda()
        #Stores the information of other queried location in a batch
        loc_query_batch = torch.zeros((int(inputs.shape[0]), args.num_actions)).cuda()
        # Active Target Information representation
        act_target_label = torch.zeros((int(inputs.shape[0]), args.num_actions)).cuda()
        final_target_label = torch.zeros((int(inputs.shape[0]), args.num_actions)).cuda()
        # stores the information of previously selected grids as target 
        mask_info = torch.ones((int(inputs.shape[0]), args.num_actions)).cuda()
        #store the information about the remaining query
        query_info = torch.zeros(int(inputs.shape[0])).cuda()

        # Start an episode
        policy_loss = []; search_history = []; reward_history = [];
        travel_cost = remain_cost =25
        # Find the loss for only for the prediction policy network
        loss_static = nn.BCEWithLogitsLoss()
        action_for_vis = list()
        for step_ in range(search_budget): 
            if (step_%batch_query == 0):
                updated_search_info = search_info 
            query_remain = search_budget - step_
            # number of query left
            query_left = torch.add(query_info, query_remain).cuda()
            # action taken by agent
            logit, grid_prob_ = search_agent.forward(inputs, updated_search_info, query_left, loc_query_batch)
            grid_prob_net = grid_prob_.view(grid_prob_.size(0), -1)
            grid_prob = F.sigmoid(grid_prob_net)
            # get the prediction of target from the agents intermediate output
            policy_pred = grid_prob_net.data.clone()
            policy_pred[policy_pred<0.5] = 0.0
            policy_pred[policy_pred>=0.5] = 1.0
            policy_pred = Variable(policy_pred)
            
            ### Statistics related to prediction
            acc, tpr = utils.acc_calc(targets, policy_pred.data)
            acc_steps.append(acc)
            tpr_steps.append(tpr)
            # get the probability distribution over grids
            probs = F.softmax(logit, dim=1)
            # assign 0 probability to those grids that is already queried by agent
            mask_probs = probs * mask_info.clone()
            action_for_vis.append(mask_probs[0].detach().cpu().numpy())
            # Sample the grid that corresponds to highest probability of being target
            policy_sample = torch.argmax(mask_probs, dim=1) 
            
            ###### cost travel 
            #For uniform cost evaluation just comment out the cost travel code section
            p1, p2 = coord(int(policy_sample))
            agent_id = int(step_% batch_query)
            if len(last_visited_grid[agent_id]) == 0:
                p1_last, p2_last = coord(int(policy_sample))
                distance = abs(p1-p1_last) + abs(p2 - p2_last)
                last_visited_grid[agent_id].append((p1, p2))
                
            else:  
                p1_last = last_visited_grid[agent_id][-1][0]
                p2_last = last_visited_grid[agent_id][-1][1]
                distance = abs(p1-p1_last) + abs(p2 - p2_last)
                last_visited_grid[agent_id].append((p1, p2))
            
            remain_cost = remain_cost - distance
            p1_last, p2_last = p1, p2
            if remain_cost < 0:
                break

            ################# cost travel
            
            # compute the reward for the agent's action
            reward_update = utils.compute_reward(targets, policy_sample.data, args.beta, args.sigma)
            # get the outcome of an action in order to compute ESR/SR 
            reward_sample = utils.compute_reward_batch(targets, policy_sample.data, args.beta, args.sigma)
            # Update search info and mask info after every query
            for sample_id in range(int(inputs.shape[0])):
                # Update the search history based on the current reward 
                loc_query_batch[sample_id, int(policy_sample[sample_id].data)] = 1
                if (int(reward_sample[sample_id]) == 1):
                     search_info[sample_id, int(policy_sample[sample_id].data)] = int(reward_sample[sample_id])
                else:
                     search_info[sample_id, int(policy_sample[sample_id].data)] = -1
                # Update the mask info based on the current action taken by the agent               
                if (int(reward_update[sample_id]) == 1):
                    act_target_label[sample_id, int(policy_sample[sample_id].data)] = 1
                else: 
                    act_target_label[sample_id, int(policy_sample[sample_id].data)] = 0
                
                # update the mask info based on the current action
                mask_info[sample_id, int(policy_sample[sample_id].data)] = 0
                for out_idx in range(args.num_actions):
                    if (mask_info[sample_id, out_idx] == 0):
                        final_target_label[sample_id, out_idx] = act_target_label[sample_id, out_idx].data
                    else:
                        final_target_label[sample_id, out_idx] = grid_prob[sample_id, out_idx].data
            # store the episodic reward in the list
            reward_history.append(reward_sample)
            if (step_%batch_query == 0):
                loss_cls = loss_static(grid_prob_net.float(), final_target_label.float().cuda()) 
                # update the policy network parameters 
                optimizer.zero_grad()
                loss_cls.backward()
                optimizer.step()

        # store the log in different log file
        with open(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)+'/'+'VIS_MQ.csv','a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow([batch_idx, ] + action_for_vis)

#------------------------------------------------------------------------------------------------------#
testset = utils.get_datasetVAS_test(args.img_size, args.data_dir+'/'+args.dataset+'/', args.num_actions, args.multiclass)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# initialize the agent
agent = utils.Model_search_Arch_Adapt_batch(num_grids=args.num_actions)

# ---- Load the pre-trained model ----------------------/home/model_vas_dota64_lv_adapt_F_topk
checkpoint = torch.load(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)+"/MQ.pt")
agent.load_state_dict(checkpoint['agent'])                                                      
start_epoch = checkpoint['epoch'] + 1
print('loaded agent from %s' % args.load)

start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print('loaded agent from %s' % args.load)

# Parallelize the models if multiple GPUs available - Important for Large Batch Size to Reduce Variance model_vas_dota64_lv_adapt_F_batch
if args.parallel:
    agent = nn.DataParallel(agent)
agent.cuda()

# Update the parameters of the policy network
optimizer = optim.Adam(agent.parameters(), lr=args.lr)

# Start testing
budgets = [int(args.num_actions/3.), ]
for budget in budgets:
    test(0, budget)