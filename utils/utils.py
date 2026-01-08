import os
import torch
import torchvision.transforms as transforms
import torchvision.models as torchmodels
import torch.nn.functional as F
import numpy as np
import open_clip
from open_clip import CLIP
import json
import copy
import pdb
from dataset.dataloader import CustomDatasetFromImages, CustomDatasetFromImagesTest
from dataset.dataloader import CustomDatasetFromImagesAndGoalObjects
from dataset.dataloader import CustomDatasetFromImagesAndGoalObjectsTest
from dataset.dataloader import CustomDatasetFromImagesGoalObjectsAndChips
from dataset.dataloader import CustomDatasetFromImagesGoalObjectsAndChipsTest

device = torch.device('cpu')
if torch.cuda.is_available():
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
print (device)

def performance_stats(policies, rewards):
    # Print the performace metrics including the average reward, average number
    
    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)

    reward = rewards.mean()
    num_unique_policy = policies.sum(1).mean()
    variance = policies.sum(1).std()

    return reward, num_unique_policy, variance

def performance_stats_search(policies, rewards):
    # Print the performace metrics including the average reward, average number
    policies = torch.cat(policies, 0)
    reward = sum(rewards)
    return reward

def compute_reward(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    #print ("target:", targets.size())
    #print (policy.size())
    ## compute reward for correct search
    reward = torch.zeros(int(targets.shape[0])).to(device)
    for sample_id in range(int(targets.shape[0])):
        
        if (targets[sample_id, int(policy[sample_id])] == 1):
            reward[sample_id] = 1 #2
        else:
            reward[sample_id] = 0 #1
    #print (reward.size())
    return reward

def compute_reward_batch(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    reward = torch.zeros(int(targets.shape[0])).cuda()
    for sample_id in range(int(targets.shape[0])):
        if (targets[sample_id, int(policy[sample_id])] == 1):
            reward[sample_id] = 1
        else:
            reward[sample_id] = 0
    
    return reward

def compute_reward_greedy(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    temp_re = torch.eq(targets.cuda(), policy).long()
    temp_re[temp_re==0] = -1
    reward = torch.sum(temp_re, dim=1).unsqueeze(1).float()
    return reward

def compute_reward_test(targets, policy, beta, sigma):
    """
    Args:
        targets: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    
    
    temp_re = torch.mul(targets.cuda(), policy)
    target_found = torch.sum(temp_re)
    num_targets = torch.sum(targets.cuda())
    total_search = torch.sum(policy.cuda())
    
    return target_found, num_targets, total_search 

def acc_calc(targets, policy):
    correct = torch.sum(policy.cuda() == targets.cuda())
    total = targets.shape[0] * targets.shape[1]
    val = correct/total
    num_targets = torch.sum(targets.cuda())
    confusion_vector = policy.cuda() / targets.cuda()
    true_positives = torch.sum(confusion_vector == 1).item()
    tpr = true_positives/num_targets
    return val, tpr
 
def get_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize(img_size), #Scale
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_size), #Scale
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform_train, transform_test

def get_transforms_clip(img_size):
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    transform_train = transforms.Compose([
        transforms.Resize(img_size), #Scale
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_size), #Scale
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform_train, transform_test

def get_datasetVLAS(img_size, root, num_grid, multiclass):
    transform_train, transform_test = get_transforms_clip(img_size)
    if not multiclass:
        trainset = CustomDatasetFromImagesAndGoalObjects(root, transform_train, num_grid)
        testset = CustomDatasetFromImagesAndGoalObjectsTest(root, transform_test, num_grid)
    else:
        trainset = CustomDatasetFromImagesAndGoalObjects(root, transform_train, num_grid, '_m')
        testset = CustomDatasetFromImagesAndGoalObjectsTest(root, transform_test, num_grid, '_m')
    return trainset, testset

def get_datasetPAGE(img_size, root, num_grid, multiclass):
    transform_train, transform_test = get_transforms_clip(img_size)
    if not multiclass:
        trainset = CustomDatasetFromImagesGoalObjectsAndChips(root, transform_train, num_grid)
        testset = CustomDatasetFromImagesGoalObjectsAndChipsTest(root, transform_test, num_grid)
    else:
        trainset = CustomDatasetFromImagesGoalObjectsAndChips(root, transform_train, num_grid, '_m')
        testset = CustomDatasetFromImagesGoalObjectsAndChipsTest(root, transform_test, num_grid, '_m')
    return trainset, testset

def get_datasetVAS(img_size, root, num_grid):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root, transform_train, num_grid)
    testset = CustomDatasetFromImagesTest(root, transform_test, num_grid)
    return trainset, testset

def get_datasetVAS_test(img_size, root, num_grid, multiclass=False):
    transform_train, transform_test = get_transforms(img_size)
    if not multiclass:
        testset = CustomDatasetFromImagesAndGoalObjectsTest(root, transform_test, num_grid)
    else:
        testset = CustomDatasetFromImagesAndGoalObjectsTest(root, transform_test, num_grid, '_m')
    return testset

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(num_output):
    agent = torchmodels.resnet50(pretrained=True)
    set_parameter_requires_grad(agent, False)
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, num_output)
    return agent

# feat ext
class Feat_Ext(torch.nn.Module):
    def __init__(self):
        super(Feat_Ext, self).__init__()
        resnet_embedding_sz = 2048
        pointwise_in_channels = 60 
        ## Input feature extractor
        res50_model = torchmodels.resnet50(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res50_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        #self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, 49, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        # Input feature extraction
        feat_ext = self.agent(x)
        reduced_feat = self.maxpool(feat_ext)  #apply maxpool stride = 2
        return reduced_feat

################# VAS #################
class Model_search_Arch(torch.nn.Module):
    def __init__(self, num_grids):
        super(Model_search_Arch, self).__init__()
        resnet_embedding_sz = 2048
        pointwise_in_channels = 60 
        res50_model = torchmodels.resnet50(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res50_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, num_grids, 1)  
        self.pointwise = torch.nn.Conv2d(int(2*num_grids), 3, 1, 1)    
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(588, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_grids),   
        )

    def forward(self, x, search_info):
        feat_ext = self.agent(x)
        reduced_feat =  F.relu(self.conv1(feat_ext))
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 14, 14)
        
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)
        
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        out = combined_feat.view(combined_feat.size(0), -1)
        
        logits = self.linear_relu_stack(out)
        return logits


################# PSVAS #################
class Model_search_Arch_Adapt(torch.nn.Module):
    def __init__(self, num_grids):
        super(Model_search_Arch_Adapt, self).__init__()
        resnet_embedding_sz = 2048
        pointwise_in_channels = 60 
        ## Input feature extractor
        res50_model = torchmodels.resnet50(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res50_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, num_grids, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.pointwise = torch.nn.Conv2d(int(2*num_grids), 3, 1, 1) #61
        
        self.pointwise_search_info = torch.nn.Conv2d(num_grids, 1, 1, 1)
        #self.pointwise_pred = torch.nn.Conv2d(98, 3, 1, 1) #60
        
        # final MLP layer to transform combine representation to action space for grid prob
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(147, 128),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_grids),    #60
        )
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(int(2*num_grids), int(1.5*num_grids)),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(int(1.5*num_grids), num_grids),    #60
        )
        self.side_length = int(num_grids ** 0.5)

    def forward(self, x, search_info, query_left):
        # Input feature extraction
        feat_ext = self.agent(x)

        # feature squezing using 1x1 conv
        reduced_feat_resnet =  F.relu(self.conv1(feat_ext))
        reduced_feat = self.maxpool(reduced_feat_resnet)  #apply maxpool stride = 2
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
        
        search_info_tile_search_module = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, self.side_length, self.side_length)
        
        # tile remaining query budget information (b)
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, self.side_length, self.side_length)
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)  #, query_info_tile
        
        ## apply 1x1 conv on the combined feature representation
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        ## apply 1*1 conv on the search info feature representation
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile_search_module))
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)

        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        grid_prob = logits
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, self.side_length, self.side_length)
        rl_rep = torch.cat((map_grid_prob, search_feat_map), dim=1)  #, query_info_tile
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl, grid_prob   


################# MPS-VAS #################
class Model_search_Arch_Adapt_pred(torch.nn.Module):
    def __init__(self, num_grids):
        super(Model_search_Arch_Adapt_pred, self).__init__()
        resnet_embedding_sz = 2048
        pointwise_in_channels = 60 
        ## Input feature extractor
        res50_model = torchmodels.resnet50(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res50_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, num_grids, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.pointwise = torch.nn.Conv2d(int(2*num_grids), 3, 1, 1) #61
        
        # final MLP layer to transform combine representation to action space for grid prob
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(147, 128),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_grids),    #60
        )
       
    def forward(self, x, search_info, query_left):
        # Input feature extraction
        feat_ext = self.agent(x)

        # feature squezing using 1x1 conv
        reduced_feat_resnet =  F.relu(self.conv1(feat_ext))
        reduced_feat = self.maxpool(reduced_feat_resnet)  #apply maxpool stride = 2
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
        
        # tile remaining query budget information (b)
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, 7, 7)
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)  #, query_info_tile
        
        ## apply 1x1 conv on the combined feature representation
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)
        
        #print (out.shape)
        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        grid_prob = logits
        
        return grid_prob        

class Model_search_Arch_Adapt_search_meta(torch.nn.Module):
    def __init__(self, num_grids):
        super(Model_search_Arch_Adapt_search_meta, self).__init__()
        
        
        self.pointwise_search_info = torch.nn.Conv2d(num_grids, 1, 1, 1)
        #self.pointwise_pred = torch.nn.Conv2d(98, 3, 1, 1) #60
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(int(2*num_grids), int(1.5*num_grids)), 
            torch.nn.ReLU(),
            torch.nn.Linear(int(1.5*num_grids), num_grids),
        )
        self.side_length = int(num_grids ** 0.5)

    def forward(self, x, search_info, query_left, grid_prob):
        
        
        # tile previous search history information (auxiliary state ot) 
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, self.side_length, self.side_length)
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, self.side_length, self.side_length)
        ## apply 1*1 conv on the search info feature representation
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile))
        
        rl_rep = torch.cat((map_grid_prob, search_feat_map), dim=1)  #, query_info_tile
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        #print (grid_prob.shape)
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl


################# MQ-VAS #################
class Model_search_Arch_Adapt_batch(torch.nn.Module):
    def __init__(self, num_grids):
        super(Model_search_Arch_Adapt_batch, self).__init__()
        resnet_embedding_sz = 2048
        pointwise_in_channels = 60 
        ## Input feature extractor
        res50_model = torchmodels.resnet50(pretrained=True)
        self.agent = torch.nn.Sequential(*list(res50_model.children())[:-2])
        for param in self.agent.parameters():
            param.requires_grad = False
        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, num_grids, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.pointwise = torch.nn.Conv2d(int(2*num_grids), 3, 1, 1) #61
        
        self.pointwise_search_info = torch.nn.Conv2d(num_grids, 1, 1, 1)
        
        self.pointwise_loc_query_info = torch.nn.Conv2d(num_grids, 1, 1, 1)
        
        # final MLP layer to transform combine representation to action space for grid prob
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(147, 128),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_grids),    #60
        )
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(int(3*num_grids), int(2*num_grids)),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(int(2*num_grids), num_grids),    #60
        )
        self.side_length = int(num_grids ** 0.5)

    def forward(self, x, search_info, query_left, loc_query_batch):
        # Input feature extraction
        feat_ext = self.agent(x)

        # feature squezing using 1x1 conv
        reduced_feat_resnet =  F.relu(self.conv1(feat_ext))
        reduced_feat = self.maxpool(reduced_feat_resnet)  #apply maxpool stride = 2 #64,7,7
        
        # tile previous search history information (auxiliary state ot) #64,7,7
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
        
        search_info_tile_search_module = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, self.side_length, self.side_length)
        
        # tile other queried location in the batch#64,8,8
        loc_batch_tile = loc_query_batch.view(loc_query_batch.shape[0], loc_query_batch.shape[1], 1, 1).repeat(1, 1, self.side_length, self.side_length)
        
        
        # tile remaining query budget information (b) #1,7,7
        query_info_tile = query_left.view(query_left.shape[0], 1, 1, 1).repeat(1, 1, self.side_length, self.side_length)
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile), dim=1)  #, query_info_tile  #128,7,7
        
        ## apply 1x1 conv on the combined feature representation #3,7,7
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        ## apply 1*1 conv on the search info feature representation
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile_search_module))
        
        ## apply 1*1 conv on the search info feature representation
        loc_query_batch_map = F.relu(self.pointwise_loc_query_info(loc_batch_tile))
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)
        
        #print (out.shape)
        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        grid_prob = logits
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, self.side_length, self.side_length)
        rl_rep = torch.cat((map_grid_prob, search_feat_map, loc_query_batch_map), dim=1)  #, query_info_tile
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        #print (grid_prob.shape)
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl, grid_prob     


################# VLAS #################
class Model_search_Arch_VLM(torch.nn.Module):
    def __init__(self, num_grids):
        super(Model_search_Arch_VLM, self).__init__()
        resnet_embedding_sz = 2048
        pointwise_in_channels = 60
        ## Input feature extractor
        self.agent, _, _ = open_clip.create_model_and_transforms("RN50", "openai")
        ckpt = torch.load("/home/star/.cache/clip/RemoteCLIP-RN50.pt")
        self.agent.load_state_dict(ckpt)

        # Remove the head of the visual module
        visual_modules = list(self.agent.visual.children())[:-1]
        self.agent.visual = torch.nn.Sequential(*visual_modules)
        # Freeze parameters
        for param in self.agent.parameters():
            param.requires_grad = False

        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, num_grids, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pointwise = torch.nn.Conv2d(int(3*num_grids), 3, 1, 1) #61
        self.pointwise_search_info = torch.nn.Conv2d(num_grids, 1, 1, 1)
        self.pointwise_loc_query_info = torch.nn.Conv2d(num_grids, 1, 1, 1)

        self.linear_relu_stack_text = torch.nn.Sequential(
            torch.nn.Linear(1024, 128), 
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_grids), 
        )
        
        # final MLP layer to transform combine representation to action space for grid prob
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(147, 128),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_grids),    #60
        )
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(int(3*num_grids), int(2*num_grids)),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(int(2*num_grids), num_grids),    #60
        )
        self.side_length = int(num_grids ** 0.5)

    def forward(self, x, classname_feat, search_info, loc_query_batch):
        # Input feature extraction
        image_feat = self.agent.encode_image(x)
        classname_feat = self.agent.encode_text(classname_feat)

        goal_object_feat = self.linear_relu_stack_text(classname_feat)
        goal_object_feat = goal_object_feat.view(goal_object_feat.shape[0], goal_object_feat.shape[1], 1, 1).repeat(1, 1, 7, 7)
        
        # feature squezing using 1x1 conv
        reduced_feat_resnet =  F.relu(self.conv1(image_feat))
        reduced_feat = self.maxpool(reduced_feat_resnet)  #apply maxpool stride = 2 #64,7,7
        
        # tile previous search history information (auxiliary state ot) #64,7,7
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
        
        search_info_tile_search_module = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, self.side_length, self.side_length)
        
        # tile other queried location in the batch#64,8,8
        loc_batch_tile = loc_query_batch.view(loc_query_batch.shape[0], loc_query_batch.shape[1], 1, 1).repeat(1, 1, self.side_length, self.side_length)
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile, goal_object_feat), dim=1) 
        
        ## apply 1x1 conv on the combined feature representation #3,7,7
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        ## apply 1*1 conv on the search info feature representation
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile_search_module))
        
        ## apply 1*1 conv on the search info feature representation
        loc_query_batch_map = F.relu(self.pointwise_loc_query_info(loc_batch_tile))
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)
        
        #print (out.shape) 
        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        grid_prob = logits
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, self.side_length, self.side_length)

        rl_rep = torch.cat((map_grid_prob, search_feat_map, loc_query_batch_map), dim=1)
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        #print (grid_prob.shape)
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl, grid_prob

################# PAGE-VLAS #################
class Model_search_Arch_VLM_PAGE(torch.nn.Module):
    def __init__(self, num_grids):
        super(Model_search_Arch_VLM_PAGE, self).__init__()
        resnet_embedding_sz = 2048
        pointwise_in_channels = 60
        ## Input feature extractor
        self.agent, _, _ = open_clip.create_model_and_transforms("RN50", "openai")
        ckpt = torch.load("/home/star/.cache/clip/RemoteCLIP-RN50.pt")
        self.agent.load_state_dict(ckpt)

        # Separate the head of the visual module
        visual_modules = list(self.agent.visual.children())[:-1]
        self.visual_head = list(self.agent.visual.children())[-1]
        self.agent.visual = torch.nn.Sequential(*visual_modules)
        # Freeze parameters
        for param in self.agent.parameters():
            param.requires_grad = False
        for param in self.visual_head.parameters():
            param.requires_grad = False

        # feature squezzing using 1x1 conv
        self.conv1 = torch.nn.Conv2d(resnet_embedding_sz, num_grids, 1) #30
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pointwise = torch.nn.Conv2d(int(3*num_grids), 3, 1, 1) #61
        self.pointwise_search_info = torch.nn.Conv2d(num_grids, 1, 1, 1)
        self.pointwise_loc_query_info = torch.nn.Conv2d(num_grids, 1, 1, 1)

        self.linear_relu_stack_text = torch.nn.Sequential(
            torch.nn.Linear(1024, 128), 
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_grids), 
        )

        self.linear_relu_stack_graph = torch.nn.Sequential(
            torch.nn.Linear(2048, 128), 
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_grids), 
        )
        
        # final MLP layer to transform combine representation to action space for grid prob
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(147, 128),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_grids),    #60
        )
        
        # final MLP layer to transform combine representation to action space for searching
        self.linear_relu_stack_rl = torch.nn.Sequential(
            torch.nn.Linear(int(4*num_grids), int(2*num_grids)),    #60
            torch.nn.ReLU(),
            torch.nn.Linear(int(2*num_grids), num_grids),    #60
        )
        self.fuse_scale = 0.005
        self.side_length = int(num_grids ** 0.5)

    def forward(self, x, classname_feat, chips_chosen, node_feat, edge, target_node_index, search_info, loc_query_batch):
        # Input feature extraction
        image_feat = self.agent.encode_image(x)
        classname_feat = self.agent.encode_text(classname_feat)

        # calculate text (name of class) feature
        goal_object_feat = self.linear_relu_stack_text(classname_feat)
        goal_object_feat = goal_object_feat.view(goal_object_feat.shape[0], goal_object_feat.shape[1], 1, 1).repeat(1, 1, 7, 7)
        
        # get feature of current chip
        chips_feat = self.agent.encode_image(chips_chosen)
        chips_feat = self.visual_head(chips_feat)

        # get node-index of current chip
        chips_feat_rep = chips_feat.view(chips_feat.shape[0], 1, chips_feat.shape[1]).repeat(1, node_feat.shape[1], 1)
        current_node_index = torch.argmax(torch.pairwise_distance(chips_feat_rep, node_feat), 1)

        # graph update
        for idx, cur in enumerate(current_node_index):
            node_feat[idx, cur, :] = self.fuse_scale*chips_feat[idx, :] + (1-self.fuse_scale)*node_feat[idx, cur, :]

        # calculate graph feature
        graph_feat = torch.bmm(edge.half(), node_feat.half()).float()
        cur_idx = current_node_index[..., None, None].expand(-1, -1, graph_feat.shape[2])
        cur_graph_feat = torch.gather(graph_feat, 1, cur_idx).squeeze(1)
        tar_idx = target_node_index[..., None, None].expand(-1, -1, graph_feat.shape[2])
        tar_graph_feat = torch.gather(graph_feat, 1, tar_idx).squeeze(1)
        graph_feat = torch.cat((cur_graph_feat, tar_graph_feat), dim=1) 
        graph_feat = self.linear_relu_stack_graph(graph_feat)

        # feature squezing using 1x1 conv
        reduced_feat_resnet =  F.relu(self.conv1(image_feat))
        reduced_feat = self.maxpool(reduced_feat_resnet)  #apply maxpool stride = 2 #64,7,7
        
        # tile previous search history information (auxiliary state ot) #64,7,7
        search_info_tile = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, 7, 7)
        
        search_info_tile_search_module = search_info.view(search_info.shape[0], search_info.shape[1], 1, 1).repeat(1, 1, self.side_length, self.side_length)
        
        # tile other queried location in the batch#64,8,8
        loc_batch_tile = loc_query_batch.view(loc_query_batch.shape[0], loc_query_batch.shape[1], 1, 1).repeat(1, 1, self.side_length, self.side_length)
        
        # channel-wise concatenation of input feature, auxiliary state and remainin search budget as explained in the paper
        inp_search_concat = torch.cat((reduced_feat, search_info_tile, goal_object_feat), dim=1) 
        
        ## apply 1x1 conv on the combined feature representation #3,7,7
        combined_feat = F.relu(self.pointwise(inp_search_concat))
        
        ## apply 1*1 conv on the search info feature representation
        search_feat_map = F.relu(self.pointwise_search_info(search_info_tile_search_module))
        
        ## apply 1*1 conv on the search info feature representation
        loc_query_batch_map = F.relu(self.pointwise_loc_query_info(loc_batch_tile))
        
        # flattened the final representation
        out = combined_feat.view(combined_feat.size(0), -1)
        
        #print (out.shape) 
        ## apply 2 layer MLP with relu activation to transform the combined feature representation into action space
        logits = self.linear_relu_stack(out)
        grid_prob = logits
        
        map_grid_prob = grid_prob.view(grid_prob.shape[0], 1, self.side_length, self.side_length)
        graph_feat = graph_feat.view(graph_feat.shape[0], 1, self.side_length, self.side_length)

        rl_rep = torch.cat((map_grid_prob, search_feat_map, loc_query_batch_map, graph_feat), dim=1) 
        rl_rep_final = rl_rep.view(rl_rep.size(0), -1)
        #print (grid_prob.shape)
        
        logits_rl = self.linear_relu_stack_rl(rl_rep_final)
        return logits_rl, grid_prob, node_feat
