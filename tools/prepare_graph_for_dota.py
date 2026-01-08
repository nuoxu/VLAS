# create two csv files with image name and label at each row
import os
import sys
sys.path.insert(0, '.')
import utils.wv_util as wv
import csv
import numpy as np
import random
import time
import pdb
import torch
import open_clip
from PIL import Image

root_path = "dataset/DOTA/"
data_path = root_path + "images/"
train_label_path = root_path + "train_labels/"

labels = {}; labels_n = {}
with open('dataset/dota_class_labels.txt') as f:
    for row in csv.reader(f):
        labels[int(row[0].split(":")[0])] = row[0].split(":")[1]
        labels_n[row[0].split(":")[1]] = int(row[0].split(":")[0])

chip_name_list = list(); label_list = list(); label_num_list = list()
input_path_list = os.listdir(data_path)
train_coords, train_chips, train_classes = wv.get_labels_from_txt(train_label_path, labels_n)
side_length = [6,7,8,9,10]


def parse_data(side_length):

    for img in input_path_list: 
        if img in train_chips:
            # Load an image
            chip_name_withdp = data_path + img
            chip_name = img
            arr = wv.get_image(chip_name_withdp)
            h = arr.shape[0]
            w = arr.shape[1]
            if len(arr.shape) == 2:
                arr = np.stack([arr, arr, arr],axis=-1)
            if len(arr.shape) == 3 and arr.shape[-1] > 3:
                arr = arr[:, :, :3]
            chip_shape = (int(np.floor(w/side_length)), int(np.floor(h/side_length)))
        
            # We only want to coordinates and classes that are within our chip
            coords = train_coords[train_chips==chip_name]
            classes = train_classes[train_chips==chip_name].astype(np.int64)
    
            # chip the image
            c_img, c_box, c_cls = wv.chip_image(img=arr, coords=coords, classes=classes, shape=chip_shape)
            print("Num Chips: %d" % c_img.shape[0], c_img.shape[1], c_img.shape[2], c_img.shape[3])
    
            label_vector = np.zeros(int(len(labels)))
            for ck in c_cls.keys():
                for idx, lk in enumerate(labels.keys()):
                    if lk in c_cls[ck]:
                        label_vector[idx] = np.sum(1.*(c_cls[ck]==lk))

                label_list.append(1.*(label_vector>0))
                label_num_list.append(label_vector)

                label_vector = np.zeros(int(len(labels)))
            chip_name_list.append(chip_name)

    return chip_name_list, label_list, label_num_list

def k_means_init(data_input, num_clusters):
    centers = []
    for i in range(num_clusters):
        i = np.random.random(np.size(data_input[0]))
        centers.append(i)
    return centers

def classify(data_input, centers, num_clusters):
    clusters = [[] for i in range(num_clusters)]
    num_list = [i for i in range(num_clusters)]
    records  =  [[] for i in range(num_clusters)]
    for label, i in enumerate(data_input):
        distants = []
        for center in centers:
            distant = np.linalg.norm(i-center)+1/(np.sum(i*center)+0.1)
            distants.append(distant)
        min_distant  =  min(distants)
        for j, distant in enumerate(distants):
            if distant == min_distant:
                clusters[j].append(i)
                records[j].append(label)
                break
    records_dict = dict(zip(num_list, records))
    return clusters, records_dict

def centers_refresh(clusters, data_input):
    centers = []
    for cluster in clusters:
        if len(cluster):
            center = sum(cluster) / len(cluster)
        else:
            center  = np.random.random(np.size(data_input[0]))
        centers.append(center)
    return centers

def judge(centers, pre_centers):
    for i in range(len(centers)):
        if (centers[i] == pre_centers[i]).all():
            pass
        else:
            return False
    return  True

def k_means(data_input, num_clusters, iter_number):
    centers = k_means_init(data_input, num_clusters)
    for i in range(iter_number):
        print("Iter Number: %s / %s, Time: %s" % (i+1, iter_number, time.asctime(time.localtime(time.time()))))
        pre_centers = centers
        clusters, records = classify(data_input, centers, num_clusters)
        centers = centers_refresh(clusters, data_input)
        if judge(centers, pre_centers):
            centers = np.array(centers)
            return records, centers
    centers = np.array(centers)
    return records, centers


def get_node_label(data_input, num_clusters, iter_number=1000):
    records, centers = k_means(data_input, num_clusters, iter_number)
    return centers

def get_node_id_for_data(data_input, centers):
    data_node_id_list = []
    for data in data_input:
        distants = []
        for center in centers:
            distant = np.linalg.norm(data-center)
            distants.append(distant)
        data_node_id_list.append(distants.index(min(distants)))
    return data_node_id_list

def get_edge(data_input, centers, side_length, iter_number=50):
    num_clusters = len(centers)
    relationships = np.zeros((num_clusters, num_clusters))
    data_node_id_list = get_node_id_for_data(data_input, centers)

    grid_num = int(side_length**2)

    for idx, dn in enumerate(data_node_id_list):
        img_id, grid_id = divmod(idx, grid_num)
        row_id = grid_id % side_length

        if row_id < side_length - 1:
            relationships[dn][data_node_id_list[int(idx+1)]] += 1

        if grid_id < grid_num - side_length:
            relationships[dn][data_node_id_list[int(idx+side_length)]] += 1

    rs = np.log(relationships + relationships.T + 1)
    
    # Sinkhorn
    for _ in range(iter_number):
        rs = ((rs.T / np.sum(rs.T, 0)).T + (rs / np.sum(rs, 0))) / 2.

    return data_node_id_list, rs

def get_node_feature(chip_name_list, num_clusters, data_node_id_list, side_length):

    node_feature = {i:[] for i in range(num_clusters)}
    grid_num = int(side_length**2)

    for idx, img in enumerate(chip_name_list):
        # Load an image
        chip_name_withdp = data_path + img
        chip_name = img
        arr = wv.get_image(chip_name_withdp)
        h = arr.shape[0]
        w = arr.shape[1]
        if len(arr.shape) == 2:
            arr = np.stack([arr, arr, arr],axis=-1)
        if len(arr.shape) == 3 and arr.shape[-1] > 3:
            arr = arr[:, :, :3]
        chip_shape = (int(np.floor(w/side_length)), int(np.floor(h/side_length)))
        
        # We only want to coordinates and classes that are within our chip
        coords = train_coords[train_chips==chip_name]
        classes = train_classes[train_chips==chip_name].astype(np.int64)
    
        # chip the image
        c_img, c_box, c_cls = wv.chip_image(img=arr, coords=coords, classes=classes, shape=chip_shape)
        print("Num Chips: %d" % c_img.shape[0], c_img.shape[1], c_img.shape[2], c_img.shape[3])

        image = torch.stack([preprocess(Image.fromarray(k.astype('uint8')).convert('RGB')) for k in c_img])
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image.cuda())
        image_features = image_features.cpu().numpy()

        for idy, imgf in enumerate(image_features):
            node_feature[data_node_id_list[int(idx*grid_num + idy)]].append(imgf)

    node_feature = np.array([np.mean(np.array(node_feature[k]), 0) for k in node_feature.keys()])

    return node_feature

if __name__ == "__main__":

    num_clusters = 12

    model_name = "RN50"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, "openai", device=device)

    ## creating csv file and the label file
    for sl in side_length:
        chip_name_list, label_list, label_num_list = parse_data(sl)
        node_label = get_node_label(np.array(label_list), num_clusters)
        data_node_id_list, node_edge = get_edge(np.array(label_list), node_label, sl)
        node_feature = get_node_feature(chip_name_list, num_clusters, data_node_id_list, sl)
        save_dict = dict(node_features=node_feature, node_labels=node_label, edges=node_edge)
        np.save('graph/dota_cluster{}_grid{}.npy'.format(num_clusters, int(sl**2)), save_dict)
        chip_name_list = list(); label_list = list(); label_num_list = list()
