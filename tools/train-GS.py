"""
train the Greedy Policy Network
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

from utils import utils

parser = argparse.ArgumentParser(description='PolicyNetworkTraining')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--data_dir', default='dataset/', help='data directory')
parser.add_argument('--dataset', choices=['DOTA', 'xView'], default='DOTA', help='name of dataset')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='model/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')#256
parser.add_argument('--img_size', type=int, default=448, help='PN Image Size')
parser.add_argument('--epoch_step', type=int, default=100, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=120, help='total epochs to run')
parser.add_argument('--num_workers', type=int, default=8, help='Number of Workers')
parser.add_argument('--test_epoch', type=int, default=10, help='At every N epoch test the network')
parser.add_argument('--parallel', action='store_true', default=True, help='use multiple GPUs for training')
parser.add_argument('--alpha', type=float, default=0.6, help='probability bounding factor')
parser.add_argument('--beta', type=float, default=0.1, help='Coarse detector increment')
parser.add_argument('--sigma', type=float, default=0.5, help='cost for patch use')
parser.add_argument('--num_actions', choices=[36, 49, 64, 81, 100], type=int, default=64, help='total number of action/grid')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.makedirs(args.cv_dir)
if not os.path.exists(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)):
    os.makedirs(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions))

def train(epoch):
    agent.train()
    rewards, rewards_baseline, policies = [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs = Variable(inputs)
        if not args.parallel:
            inputs = inputs.cuda()

        # Actions by the Agent
        probs = F.sigmoid(agent.forward(inputs))
        alpha_hp = np.clip(args.alpha + epoch * 0.001, 0.6, 0.95)
        probs = probs*alpha_hp + (1-alpha_hp) * (1-probs)

        # Sample the policies from the Bernoulli distribution characterized by agent
        distr = Bernoulli(probs)
        policy_sample = distr.sample()

        # Test time policy - used as baseline policy in the training step
        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)
        
        # Find the reward for baseline and sampled policy
        reward_map = utils.compute_reward_greedy(targets, policy_map.data, args.beta, args.sigma)
        reward_sample = utils.compute_reward_greedy(targets, policy_sample.data, args.beta, args.sigma)
        advantage = reward_sample.cuda().float() - reward_map.cuda().float()

        # Find the loss for only the policy network
        loss = -distr.log_prob(policy_sample)
        
        loss = loss * Variable(advantage).expand_as(policy_sample)
        
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards.append(reward_sample.cpu())
        rewards_baseline.append(reward_map.cpu())
        policies.append(policy_sample.data.cpu())

    reward, sparsity, variance = utils.performance_stats(policies, rewards)
    
    with open(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)+'/'+'GS_log.txt','a') as f:
        f.write('Train: %d | Rw: %.2f \n' % (epoch, reward))
    
    print('Train: %d | Rw: %.2f | S: %.3f | V: %.3f' % (epoch, reward, sparsity, variance))
    
    
def test(epoch):
    search_budget = int(args.num_actions/3.)
    agent.eval()
    targets_found, metrics, policies, set_labels, num_targets, num_search = list(), [], [], [], list(), list()
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

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

        targets_, total_targets, total_search = utils.compute_reward_test(targets, policy.data, args.beta, args.sigma)
        
        for sample_id in range(int(inputs.shape[0])):
            temp = int(torch.sum(targets[sample_id,:]))
            if (temp > search_budget):
                num_targets.append(search_budget)
            else:
                num_targets.append(temp)

        targets_found.append(targets_)
        
    
    recall = sum(targets_found) / sum(num_targets)
    
    with open(args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)+'/'+'GS_log.txt','a') as f:
        f.write('Test - Recall: %.2f \n' % (recall))
    
    print('Test - Recall: %.2f | SB: %.2f' % (recall,search_budget))
    
    # save the model --- agent
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
    state = {
      'agent': agent_state_dict,
      'epoch': epoch,
      'reward': recall,
    }
    torch.save(state, args.cv_dir+'/'+args.dataset+'/'+str(args.num_actions)+"/GS.pt")

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_datasetVAS(args.img_size, args.data_dir+'/'+args.dataset+'/', args.num_actions)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = torchdata.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_workers)

agent = utils.get_model(args.num_actions)

# ---- Load the pre-trained model ----------------------
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
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    if epoch % args.test_epoch == 0:
        test(epoch)
