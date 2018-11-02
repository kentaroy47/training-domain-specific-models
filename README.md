# training-domain-specific-models
This is a framework to train domain specific object detection models.

Faster-RCNN implementation is based on faster-rcnn.pytorch by jwyang. Thanks!

I strongly recommend to take a look at their readme if you get stuck on frcnn codes.


https://github.com/jwyang/faster-rcnn.pytorch

# What is a domain specific model?
For a backbone of object detection, Resnet101 is a very good model but TOO BIG!

A domain specific model(DSM) is a model focusing on achieving high accuracy
at a limited domain (e.g. fixed view of an intersection). We argue that DSMs
can capture essential features well even with a small model size.

In this repo, we train a small domain specific model (say res18) in with a dataset of a limited domain.

We see that by training, small models can achieve very high accuracy!

![dsm](https://github.com/kentaroy47/training-domain-specific-models/blob/master/fig1_v2.jpg)


# Preparation

## Requirements.
Pytorch 0.4.0

Python 3.x

CUDA 8.0 or higher

## Clone repo
Lets start off by cloning this repo.

```
git clone https://github.com/kentaroy47/training-domain-specific-models.git
cd training-domain-specific-models
```

You may need to compile the rpn scripts.

Please see jwyang's repo for details.

https://github.com/jwyang/faster-rcnn.pytorch

## Download models
We need to prepare Resnet101 and Resnet18 Faster-RCNN model.

```
cd training-domain-specific-models
mkdir models/
cd models

# resnet101 COCO trained model. This is from faster-rcnn.pytorch repo.
Download.. https://www.dropbox.com/s/dpq6qv0efspelr3/faster_rcnn_1_10_9771.pth?dl=0

# resnet18 COCO trained model
Download.. https://drive.google.com/file/d/1KvrBMDYD5QtccjWbeKsLDZj6gBYRwVum/view?usp=sharing
cd ..
```

## Download dataset
We release two survillance videos you can test on. (coral.mp4 is from noscope)
Here, we train domain specific model on such domain.

```
Download..  https://drive.google.com/file/d/1TnNcOpLqJzBwqYfRs7Oh8WRHOlk6I5ET/view?usp=sharing

plz extract in the repo dir.
tar -zxvf images.tar.gz

```

## Setup dataset
If the models and the video are set, we can prepare the dataset.

1. Res101 model generates the teacher labels.
2. The dataset is prepared in a PASCAL_VOC format for training.

This is done in a single script.

Just run:

```
# for dataset coral
python make_dataset.py　--dataset coral
# for dataset jackson2
python make_dataset.py　--dataset jackson2
```

### shortcut..
We prepared a dataset.tar in the link bellow, if you want to take a short cut.

Actually cloning the repo will get you the pickle lable files (output/baseline/)

# Training Domain Specific Models!
Run..

This will take about 2 hours on TitanXp.

```

python trainval_net_ds.py --cuda --r True --dataset pascal_voc_jackson2

# or for coral,
python trainval_net_ds.py --cuda --r True --dataset pascal_voc_coral

```

# Evaluation!
We evaluate the accuracy (mAP) with validation images.

The res101 outputs are utilized as ground truth here, since labeling them are cubersome.

```
python demo-and-eval-save.py --cuda --dataset pascal_voc_jackson2 --image_dir images/jackson2_val
```
