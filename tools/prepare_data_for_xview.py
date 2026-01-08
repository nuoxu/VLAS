# create two csv files with image name and label at each row
import os
import sys
sys.path.insert(0, '.')
import utils.wv_util as wv
import csv
import numpy as np
import random
np.set_printoptions(suppress=True)


root_path = "dataset/xView/"
data_path = root_path + "images/"
label_path = root_path + "xView_train.geojson"
train_chip_name_list = list(); train_label_list = list(); train_class_name_list = list(); train_label_num_list = list()
val_chip_name_list = list(); val_label_list = list(); val_class_name_list = list(); val_label_num_list = list()
input_path_list = os.listdir(data_path)
all_coords, all_chips, all_classes = wv.get_labels_from_geojson(label_path)
side_length = [6,7,8,9,10]

labels = {}
with open('dataset/xview_class_labels.txt') as f:
    for row in csv.reader(f):
        labels[int(row[0].split(":")[0])] = row[0].split(":")[1]


def split(val_rate=0.3):
    return random.sample(input_path_list, int(len(input_path_list)*val_rate))

def category_distribution(val_list):
    category_d = {k:0 for k in labels.keys()}
    for img in input_path_list: 
        classes = all_classes[all_chips==img].astype(np.int64)
        for cs in classes:
            if cs in labels.keys():
                category_d[cs] += 1
    return category_d


def parse_data(val_list, side_length):

    for img in input_path_list: 
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

        # Assign label to each chip
        if img in val_list:
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
        else:
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

    with open(root_path + 'train_images_grid{}.csv'.format(int(side_length ** 2)), 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for (img_name, img_label, class_name, label_num) in zip(train_chip_name_list, train_label_list, train_class_name_list, train_label_num_list):
            filewriter.writerow([img_name, img_label, class_name, label_num])

    with open(root_path + 'val_images_grid{}.csv'.format(int(side_length ** 2)), 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for (img_name, img_label, class_name, label_num) in zip(val_chip_name_list, val_label_list, val_class_name_list, val_label_num_list):
            filewriter.writerow([img_name, img_label, class_name, label_num])


if __name__ == "__main__":

    ## split training dataset and test dataset
    # val_list = split(0.3)
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
    category_d = category_distribution(val_list)
    print(category_d)
    print(labels)

    ## creating csv file and the label file
    for sl in side_length:
        parse_data(val_list, sl)
        train_chip_name_list = list(); train_label_list = list(); train_class_name_list = list(); train_label_num_list = list()
        val_chip_name_list = list(); val_label_list = list(); val_class_name_list = list(); val_label_num_list = list()
