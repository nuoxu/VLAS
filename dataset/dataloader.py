import pandas as pd
import numpy as np
import warnings
import torch
import cv2

from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

####FixMatch
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import pdb
import open_clip


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    #assert -30 <= v <= 30
    #if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)



def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert v >= 0.0
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert 0 <= v
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):  #[0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

    
def augment_list():  
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95)
    ]
    return l

def chip_img(img,shape=(300,300)):
    """
    Chip an image.

    Args:
        img: the image to be chipped in array format
        shape: an (W,H) tuple indicating width and height of chips

    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.
    """
    height = img.shape[0]
    width = img.shape[1]
    wn,hn = shape
    
    w_num,h_num = (int(width/wn),int(height/hn))
    images = np.zeros((w_num*h_num,hn,wn,3))
    
    k = 0
    for i in range(w_num):
        for j in range(h_num):
            
            chip = img[hn*j:hn*(j+1),wn*i:wn*(i+1),:]
            images[k]=chip
            
            k = k + 1
    
    return images.astype(np.uint8)
    
class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

        
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            img = op(img, val) 
        cutout_val = random.random() * 0.5 
        img = Cutout(img, cutout_val) #for fixmatch
        return img


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform, num_grid):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        self.img_path = csv_path+'images/'
        # Read the csv file
        data_info = pd.read_csv(csv_path+'train_images_grid{}.csv'.format(num_grid), header=None)
        self.image_arr = np.asarray(data_info.iloc[:, 0])
        self.label_arr = np.asarray(data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.image_arr) #data_info

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        #img = cv2.imread(single_image_name)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img_as_img = Image.fromarray(img)
        
        img_as_img = Image.open(self.img_path+single_image_name).convert('RGB') #.convert('RGB') for dota
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        single_image_label = single_image_label.replace('\n', '')[1:-2].split('. ')
        single_image_label = np.array([float(sil) for sil in single_image_label])
        single_image_label = torch.from_numpy(single_image_label)

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

class CustomDatasetFromImagesAndGoalObjects(Dataset):
    def __init__(self, csv_path, transform, num_grid, multiclass=''):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        self.img_path = csv_path+'images/'
        # Read the csv file
        data_info = pd.read_csv(csv_path+'train_images_grid{}{}.csv'.format(num_grid, multiclass), header=None)
        self.image_arr = np.asarray(data_info.iloc[:, 0])
        self.label_arr = np.asarray(data_info.iloc[:, 1])
        self.class_name = np.asarray(data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.image_arr) #data_info

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        
        img_as_img = Image.open(self.img_path+single_image_name).convert('RGB') #.convert('RGB') for dota
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        single_image_label = single_image_label.replace('\n', '')[1:-2].split('. ')
        single_image_label = np.array([float(sil) for sil in single_image_label])
        single_image_label = torch.from_numpy(single_image_label)

        # The name of category
        class_name = self.class_name[index]
        return (img_as_tensor, single_image_label, class_name)

    def __len__(self):
        return self.data_len

class CustomDatasetFromImagesGoalObjectsAndChips(Dataset):
    def __init__(self, csv_path, transform, num_grid, multiclass=''):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        self.img_path = csv_path+'images/'
        # Read the csv file
        data_info = pd.read_csv(csv_path+'train_images_grid{}{}.csv'.format(num_grid, multiclass), header=None)
        self.image_arr = np.asarray(data_info.iloc[:, 0])
        self.label_arr = np.asarray(data_info.iloc[:, 1])
        self.class_name = np.asarray(data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.image_arr) #data_info
        self.side_length = int(np.sqrt(num_grid))
        _, _, self.preprocess = open_clip.create_model_and_transforms("RN50", "openai")

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        
        img_as_img = Image.open(self.img_path+single_image_name).convert('RGB') #.convert('RGB') for dota
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Chip the image
        img_as_numpy = np.array(img_as_img)
        h, w = img_as_numpy.shape[:2]
        chip_shape = (int(np.floor(w/self.side_length)), int(np.floor(h/self.side_length)))
        img_as_chips = chip_img(img_as_numpy, chip_shape)
        img_as_chips_feat = torch.stack([self.preprocess(Image.fromarray(k.astype('uint8')).convert('RGB')) for k in img_as_chips])

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        single_image_label = single_image_label.replace('\n', '')[1:-2].split('. ')
        single_image_label = np.array([float(sil) for sil in single_image_label])
        single_image_label = torch.from_numpy(single_image_label)

        # The name of category
        class_name = self.class_name[index]
        return (img_as_tensor, single_image_label, class_name, img_as_chips_feat)

    def __len__(self):
        return self.data_len

class CustomDatasetFromImagesTest(Dataset):
    def __init__(self, csv_path, transform, num_grid):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        self.img_path = csv_path+'images/'
        # Read the csv file
        data_info = pd.read_csv(csv_path+'val_images_grid{}.csv'.format(num_grid), header=None)
        self.image_arr = np.asarray(data_info.iloc[:, 0])
        self.label_arr = np.asarray(data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(self.img_path+single_image_name).convert('RGB') #.convert('RGB') for dota
        # save a image using extension
        #im1 = img_as_img.save("img.jpg")
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        single_image_label = single_image_label.replace('\n', '')[1:-2].split('. ')
        single_image_label = np.array([float(sil) for sil in single_image_label])
        single_image_label = torch.from_numpy(single_image_label)

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

class CustomDatasetFromImagesAndGoalObjectsTest(Dataset):
    def __init__(self, csv_path, transform, num_grid, multiclass=''):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        self.img_path = csv_path+'images/'
        # Read the csv file
        data_info = pd.read_csv(csv_path+'val_images_grid{}{}.csv'.format(num_grid, multiclass), header=None)
        self.image_arr = np.asarray(data_info.iloc[:, 0])
        self.label_arr = np.asarray(data_info.iloc[:, 1])
        self.class_name = np.asarray(data_info.iloc[:, 2])
        self.label_num = np.asarray(data_info.iloc[:, 3])
        # Calculate len
        self.data_len = len(self.image_arr)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(self.img_path+single_image_name).convert('RGB') #.convert('RGB') for dota
        # save a image using extension
        #im1 = img_as_img.save("img.jpg")
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        single_image_label = single_image_label.replace('\n', '')[1:-2].split('. ')
        single_image_label = np.array([float(sil) for sil in single_image_label])
        single_image_label = torch.from_numpy(single_image_label)

        # The name of category
        class_name = self.class_name[index]
        # The number of objects of a certain category in each grid
        single_image_label_number = self.label_num[index]
        single_image_label_number = single_image_label_number.replace('\n', '')[1:-2].split('. ')
        single_image_label_number = np.array([float(sil) for sil in single_image_label_number])
        single_image_label_number = torch.from_numpy(single_image_label_number)

        return (img_as_tensor, single_image_label, class_name, single_image_label_number)

    def __len__(self):
        return self.data_len

class CustomDatasetFromImagesGoalObjectsAndChipsTest(Dataset):
    def __init__(self, csv_path, transform, num_grid, multiclass=''):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        self.img_path = csv_path+'images/'
        # Read the csv file
        data_info = pd.read_csv(csv_path+'val_images_grid{}{}.csv'.format(num_grid, multiclass), header=None)
        self.image_arr = np.asarray(data_info.iloc[:, 0])
        self.label_arr = np.asarray(data_info.iloc[:, 1])
        self.class_name = np.asarray(data_info.iloc[:, 2])
        self.label_num = np.asarray(data_info.iloc[:, 3])
        # Calculate len
        self.data_len = len(self.image_arr)
        self.side_length = int(np.sqrt(num_grid))
        _, _, self.preprocess = open_clip.create_model_and_transforms("RN50", "openai")

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(self.img_path+single_image_name).convert('RGB') #.convert('RGB') for dota
        # save a image using extension
        #im1 = img_as_img.save("img.jpg")
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)

        # Chip the image
        img_as_numpy = np.array(img_as_img)
        h, w = img_as_numpy.shape[:2]
        chip_shape = (int(np.floor(w/self.side_length)), int(np.floor(h/self.side_length)))
        img_as_chips = chip_img(img_as_numpy, chip_shape)
        img_as_chips_feat = torch.stack([self.preprocess(Image.fromarray(k.astype('uint8')).convert('RGB')) for k in img_as_chips])

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        single_image_label = single_image_label.replace('\n', '')[1:-2].split('. ')
        single_image_label = np.array([float(sil) for sil in single_image_label])
        single_image_label = torch.from_numpy(single_image_label)

        # The name of category
        class_name = self.class_name[index]
        # The number of objects of a certain category in each grid
        single_image_label_number = self.label_num[index]
        single_image_label_number = single_image_label_number.replace('\n', '')[1:-2].split('. ')
        single_image_label_number = np.array([float(sil) for sil in single_image_label_number])
        single_image_label_number = torch.from_numpy(single_image_label_number)

        return (img_as_tensor, single_image_label, class_name, single_image_label_number, img_as_chips_feat)

    def __len__(self):
        return self.data_len