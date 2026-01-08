import os
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import csv
import pdb


def change_format(text):
    text = text.replace('\n', '')[1:-2].split('. ')
    text = np.array([float(sil) for sil in text])
    return text

def single_to_multi(csv_path):
    data_info = pd.read_csv(csv_path, header=None)
    image_arr = np.asarray(data_info.iloc[:, 0])
    label_arr = np.asarray(data_info.iloc[:, 1])
    class_name = np.asarray(data_info.iloc[:, 2])
    label_num = np.asarray(data_info.iloc[:, 3])

    chip_name_list = list()
    label_list = list()
    class_name_list = list()
    label_num_list = list()

    for idx in range(len(image_arr)):
        chip_name_list.append(image_arr[idx])
        label_list.append(change_format(label_arr[idx]))
        class_name_list.append(class_name[idx])
        label_num_list.append(change_format(label_num[idx]))

        for jdx in range(idx+1, len(image_arr)):
            if image_arr[idx] == image_arr[jdx]:
                ln = change_format(label_num[idx]) + change_format(label_num[jdx])
                chip_name_list.append(image_arr[idx])
                label_list.append(1.*(ln>0))
                class_name_list.append(class_name[idx]+', '+class_name[jdx])
                label_num_list.append(ln)

    return chip_name_list, label_list, class_name_list, label_num_list

if __name__ == "__main__":
    dataset_name = "xView"
    dir_path = "/home/star/hard_disk/" + dataset_name
    csv_list = os.listdir(dir_path)

    for csv_file in csv_list:
        csv_fl = os.path.splitext(csv_file)

        if csv_fl[1] == '.csv':
            csv_path = os.path.join(dir_path, csv_file)
            chip_name_list, label_list, class_name_list, label_num_list = single_to_multi(csv_path)

            with open(os.path.join(dir_path, csv_fl[0]+'_m'+csv_fl[1]), 'w') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',')
                for (img_name, img_label, class_name, label_num) in zip(chip_name_list, label_list, class_name_list, label_num_list):
                    filewriter.writerow([img_name, img_label, class_name, label_num])
