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

root_path = "dataset/xView/"
data_path = root_path + "images/"
label_path = root_path + "xView_train.geojson"
chip_name_list = list(); label_list = list(); label_num_list = list()
input_path_list = os.listdir(data_path)
all_coords, all_chips, all_classes = wv.get_labels_from_geojson(label_path)
side_length = [6,7,8,9,10]

labels = {}
with open('dataset/xview_class_labels.txt') as f:
    for row in csv.reader(f):
        labels[int(row[0].split(":")[0])] = row[0].split(":")[1]

def parse_data(val_list, side_length):

    for img in input_path_list:
        if img not in val_list:
            # Load an image
            chip_name_withdp = data_path + img
            chip_name = img
            arr = wv.get_image(chip_name_withdp)
            h, w = arr.shape[:2]
            chip_shape = (int(np.floor(w/side_length)), int(np.floor(h/side_length)))
        
            # We only want to coordinates and classes that are within our chip
            coords = all_coords[all_chips==chip_name]
            classes = all_classes[all_chips==chip_name].astype(np.int64)
    
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
        h, w = arr.shape[:2]
        chip_shape = (int(np.floor(w/side_length)), int(np.floor(h/side_length)))
        
        # We only want to coordinates and classes that are within our chip
        coords = all_coords[all_chips==chip_name]
        classes = all_classes[all_chips==chip_name].astype(np.int64)
    
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

# import networkx as nx
# def get_graph(num_clusters, node_label, node_feature, node_edge):
#     graph = nx.Graph()
#     for i in range(num_clusters):
#         graph.add_node(i, feature=node_feature[i], label=node_label[i])
#         for j in range(i, num_clusters):
#             graph.add_edge(i, j, weight=node_edge[i][j])
#     return graph

if __name__ == "__main__":

    ## split training dataset and test dataset
    val_list = ['931.tif', '112.tif', '2504.tif', '568.tif', '1244.tif', '711.tif', '1740.tif', '1883.tif', 
                '237.tif', '2250.tif', '815.tif', '1379.tif', '2516.tif', '2469.tif', '1406.tif', '2413.tif', 
                '531.tif', '1980.tif', '1858.tif', '863.tif', '2207.tif', '102.tif', '445.tif', '1182.tif', 
                '862.tif', '2513.tif', '2552.tif', '548.tif', '612.tif', '2264.tif', '2010.tif', '217.tif', 
                '521.tif', '609.tif', '546.tif', '805.tif', '506.tif', '1106.tif', '1985.tif', '537.tif', 
                '1838.tif', '669.tif', '1585.tif', '1196.tif', '323.tif', '399.tif', '2530.tif', '716.tif', 
                '1505.tif', '2591.tif', '2538.tif', '1246.tif', '1692.tif', '2031.tif', '1072.tif', '1280.tif', 
                '2561.tif', '735.tif', '791.tif', '216.tif', '822.tif', '1832.tif', '486.tif', '1848.tif', 
                '1741.tif', '193.tif', '1090.tif', '107.tif', '2462.tif', '1465.tif', '606.tif', '423.tif', 
                '1425.tif', '2399.tif', '2564.tif', '1046.tif', '2017.tif', '1929.tif', '1821.tif', '2503.tif', 
                '2542.tif', '1919.tif', '627.tif', '1309.tif', '1651.tif', '1587.tif', '1085.tif', '1154.tif', 
                '819.tif', '1880.tif', '1654.tif', '2355.tif', '672.tif', '463.tif', '2619.tif', '1922.tif', 
                '575.tif', '802.tif', '1139.tif', '2128.tif', '2568.tif', '1886.tif', '768.tif', '1311.tif', 
                '1284.tif', '2515.tif', '389.tif', '1855.tif', '1454.tif', '1445.tif', '914.tif', '2486.tif', 
                '2053.tif', '1823.tif', '1109.tif', '2251.tif', '159.tif', '104.tif', '1702.tif', '2193.tif', 
                '1459.tif', '2044.tif', '38.tif', '1051.tif', '2004.tif', '111.tif', '1945.tif', '1508.tif', 
                '2550.tif', '1351.tif', '2225.tif', '2239.tif', '1212.tif', '720.tif', '2475.tif', '708.tif', 
                '595.tif', '1065.tif', '734.tif', '1900.tif', '2334.tif', '377.tif', '1120.tif', '1834.tif', 
                '1690.tif', '128.tif', '905.tif', '2021.tif', '375.tif', '1452.tif', '320.tif', '457.tif', 
                '2313.tif', '871.tif', '2011.tif', '2159.tif', '1118.tif', '910.tif', '561.tif', '2423.tif', 
                '1920.tif', '1896.tif', '109.tif', '46.tif', '1831.tif', '619.tif', '1353.tif', '620.tif', 
                '84.tif', '639.tif', '1216.tif', '2524.tif', '665.tif', '727.tif', '181.tif', '1403.tif', 
                '2269.tif', '2353.tif', '893.tif', '2543.tif', '2398.tif', '1124.tif', '2596.tif', '254.tif', 
                '2247.tif', '382.tif', '455.tif', '136.tif', '692.tif', '959.tif', '1374.tif', '563.tif', 
                '960.tif', '373.tif', '2485.tif', '1784.tif', '1593.tif', '289.tif', '569.tif', '510.tif', 
                '451.tif', '859.tif', '860.tif', '149.tif', '2114.tif', '535.tif', '252.tif', '95.tif', 
                '1881.tif', '1630.tif', '1806.tif', '1472.tif', '541.tif', '31.tif', '372.tif', '1438.tif', 
                '2544.tif', '106.tif', '386.tif', '2009.tif', '1770.tif', '1180.tif', '1600.tif', '658.tif', 
                '18.tif', '1245.tif', '584.tif', '2310.tif', '559.tif', '598.tif', '1378.tif', '2215.tif', 
                '637.tif', '895.tif', '73.tif', '433.tif', '928.tif', '90.tif', '629.tif', '1135.tif', 
                '469.tif', '2509.tif', '2106.tif', '317.tif', '434.tif', '741.tif', '1908.tif', '1940.tif', 
                '1056.tif', '1972.tif', '92.tif', '1439.tif', '2279.tif']

    num_clusters = 36

    model_name = "RN50"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, "openai", device=device)

    ## creating csv file and the label file
    for sl in side_length:
        chip_name_list, label_list, label_num_list = parse_data(val_list, sl)
        node_label = get_node_label(np.array(label_list), num_clusters)
        data_node_id_list, node_edge = get_edge(np.array(label_list), node_label, sl)
        node_feature = get_node_feature(chip_name_list, num_clusters, data_node_id_list, sl)
        save_dict = dict(node_features=node_feature, node_labels=node_label, edges=node_edge)
        np.save('graph/xview_cluster{}_grid{}.npy'.format(num_clusters, int(sl**2)), save_dict)
        chip_name_list = list(); label_list = list(); label_num_list = list()
