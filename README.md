# PlanarReconstruction

PyTorch implementation of our CVPR 2019 paper:

[Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding](https://arxiv.org/pdf/1902.09777.pdf)

Zehao Yu\*,
[Jia Zheng](https://bertjiazheng.github.io/)\*,
[Dongze Lian](https://svip-lab.github.io/team/liandz.html),
[Zihan Zhou](https://faculty.ist.psu.edu/zzhou/Home.html),
[Shenghua Gao](http://sist.shanghaitech.edu.cn/sist_en/2018/0820/c3846a31775/page.htm)

(\* Equal Contribution)

<img src="misc/pipeline.jpg" width="800">

## Getting Started

### Installation

Clone repository and use [git-lfs](https://git-lfs.github.com/) to fetch the trained model (or download [here](https://drive.google.com/file/d/1Aa1Jb0CGpiYXKHeTwpXAwcwu_yEqdkte/view?usp=sharing)):
```bash
git clone git@github.com:svip-lab/PlanarReconstruction.git
```

We use Python 3. Create an Anaconda enviroment and install the dependencies:
```bash
conda create -y -n plane python=3.6
conda activate plane
conda install -c menpo opencv
pip install -r requirements.txt
```

### Downloading and converting data
Please download the *.tfrecords* files for training and testing converted by [PlaneNet](https://github.com/art-programmer/PlaneNet), then convert the *.tfrecords* to *.npz* files:
```bash
python data_tools/convert_tfrecords.py --data_type=train --input_tfrecords_file=/path/to/planes_scannet_train.tfrecords --output_dir=/path/to/save/processd/data
python data_tools/convert_tfrecords.py --data_type=val --input_tfrecords_file=/path/to/planes_scannet_val.tfrecords --output_dir=/path/to/save/processd/data
```

### Training
Run the following command to train our network:
```bash
python main.py train with dataset.root_dir=/path/to/save/processd/data
```

### Evaluation
Run the following command to evaluate the performance:
```bash
python main.py eval with dataset.root_dir=/path/to/save/processd/data resume_dir=/path/to/pretrained.pt dataset.batch_size=1
```

### Prediction
Run the following command to predict on a single image:
```bash
python predict.py with resume_dir=pretrained.pt image_path=/path/to/image
```

## Acknowledgements
We thank [Chen Liu](http://art-programmer.github.io/index.html) for his great works and repos.

## Citation
Please cite our paper for any purpose of usage.
```
@inproceedings{YuZLZG19,
  author    = {Zehao Yu and Jia Zheng and Dongze Lian and Zihan Zhou and Shenghua Gao},
  title     = {Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding},
  booktitle = {CVPR},
  pages     = {1029--1037},
  year      = {2019}
}
```
