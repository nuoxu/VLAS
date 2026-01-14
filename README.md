# Visual-Language Active Search for Wide-Area Remote Sensing Imagery

## Setup
- You should install prerequisites using: `pip install -r requirements.txt`
- Our settings follow previous works, please refer to [VAS](https://github.com/anindyasarkarIITH/VAS).

## Dataset
Please downoad [DOTA](https://captain-whu.github.io/DOTA/dataset.html) and [xView](https://challenge.xviewdataset.org/login) and place them in `dataset/`.
```shell
python3 tools/prepare_data_for_dota.py
```
```shell
python3 tools/prepare_data_for_xview.py
```

## Training and Evaluation
### Training 
```shell
python3 tools/train-VLAS.py \
      --dataset DOTA (or xView) \
      --cv_dir path_to_you_model \
      --num_actions 36/49/64/81/100 \
      --num_cluster 12/24/36 \
      (--multiclass)
```
```shell
python3 tools/train-PAGE.py \
      --dataset DOTA \
      --cv_dir path_to_you_model \
      --num_actions 100
```
### Evaluation
```shell
python3 tools/test-VLAS.py \
      --dataset DOTA \
      --cv_dir path_to_you_model \
      --num_actions 100
```
```shell
python3 tools/test-PAGE.py \
      --dataset DOTA \
      --cv_dir path_to_you_model \
      --num_actions 100
```
### Visualization
```shell
python3 tools/vis-PAGE.py \
      --dataset DOTA \
      --cv_dir path_to_you_model \
      --num_actions 100
```
