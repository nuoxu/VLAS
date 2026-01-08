# create two csv files with image name and label at each row
import os
import sys
sys.path.insert(0, '.')
import utils.wv_util as wv
import csv
import numpy as np
import random
np.set_printoptions(suppress=True)
import pdb

root_path = "dataset/DOTA/"
data_path = root_path + "images/"
train_label_path = root_path + "train_labels/"
val_label_path = root_path + "val_labels/"

labels = {}; labels_n = {}
with open('dataset/dota_class_labels.txt') as f:
    for row in csv.reader(f):
        labels[int(row[0].split(":")[0])] = row[0].split(":")[1]
        labels_n[row[0].split(":")[1]] = int(row[0].split(":")[0])

train_chip_name_list = list(); train_label_list = list(); train_class_name_list = list(); train_label_num_list = list()
val_chip_name_list = list(); val_label_list = list(); val_class_name_list = list(); val_label_num_list = list()
input_path_list = os.listdir(data_path)
train_coords, train_chips, train_classes = wv.get_labels_from_txt(train_label_path, labels_n)
val_coords, val_chips, val_classes = wv.get_labels_from_txt(val_label_path, labels_n)
side_length = [6,7,8,9,10]


def parse_data(side_length):

    for img in input_path_list: 
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
        if img in train_chips:
            coords = train_coords[train_chips==chip_name]
            classes = train_classes[train_chips==chip_name].astype(np.int64)
        else:
            coords = val_coords[val_chips==chip_name]
            classes = val_classes[val_chips==chip_name].astype(np.int64)

        # chip the image
        c_img, c_box, c_cls = wv.chip_image(img=arr, coords=coords, classes=classes, shape=chip_shape)
        print("Num Chips: %d" % c_img.shape[0], c_img.shape[1], c_img.shape[2], c_img.shape[3])

        # Assign label to each chip
        if img in train_chips:
            label_vector = np.zeros(int(c_img.shape[0]))
            for lk in labels.keys():
                for idx, val in enumerate(c_cls.values()):
                    if lk in val:
                        label_vector[idx] = np.sum(1.*(val==lk))
                if np.sum(label_vector) > 0:
                    train_chip_name_list.append(chip_name)
                    train_label_list.append(1.*(label_vector>0))
                    train_class_name_list.append(labels[lk])
                    train_label_num_list.append(label_vector)
                label_vector = np.zeros(int(c_img.shape[0]))
        else:
            label_vector = np.zeros(int(c_img.shape[0]))
            for lk in labels.keys():
                for idx, val in enumerate(c_cls.values()):
                    if lk in val:
                        label_vector[idx] = np.sum(1.*(val==lk)) 
                if np.sum(label_vector) > 0:
                    val_chip_name_list.append(chip_name)
                    val_label_list.append(1.*(label_vector>0))
                    val_class_name_list.append(labels[lk])
                    val_label_num_list.append(label_vector)
                label_vector = np.zeros(int(c_img.shape[0]))

    with open(root_path + 'train_images_grid{}.csv'.format(int(side_length ** 2)), 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for (img_name, img_label, class_name, label_num) in zip(train_chip_name_list, train_label_list, train_class_name_list, train_label_num_list):
            filewriter.writerow([img_name, img_label, class_name, label_num])

    with open(root_path + 'val_images_grid{}.csv'.format(int(side_length ** 2)), 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for (img_name, img_label, class_name, label_num) in zip(val_chip_name_list, val_label_list, val_class_name_list, val_label_num_list):
            filewriter.writerow([img_name, img_label, class_name, label_num])


if __name__ == "__main__":

    ## creating csv file and the label file
    for sl in side_length:
        parse_data(sl)
        train_chip_name_list = list(); train_label_list = list(); train_class_name_list = list(); train_label_num_list = list()
        val_chip_name_list = list(); val_label_list = list(); val_class_name_list = list(); val_label_num_list = list()
