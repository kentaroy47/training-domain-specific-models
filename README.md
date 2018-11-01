# training-domain-specific-models
Training domain specific object detection models.

Faster-RCNN implementation is based on faster-rcnn.pytorch by jwyang. Thanks!

https://github.com/jwyang/faster-rcnn.pytorch

# What is a domain specific model?
Resnet101 is a very good model but TOO BIG!

A domain specific model(DSM) is a model focusing on achieving high accuracy
at a limited domain (e.g. fixed view of an intersection). We argue that DSMs
can capture essential features well even with a small model size.

In this repo, we train a small domain specific model (say res18) in with a dataset of a limited domain.

We see that by training, small models can achieve very high accuracy!

![dsm](https://github.com/kentaroy47/training-domain-specific-models/blob/master/images/fig1_v2.jpg)


# Preparation

## Requirements.
Pytorch 0.4

Python 3.x

CUDA 8.0 or higher

## Clone repo
Lets start off by cloning this repo.

```
git clone https://github.com/kentaroy47/training-domain-specific-models.git
```

## Download models
We need to prepare Resnet101 and Resnet18 Faster-RCNN model.

```
cd training-domain-specific-models
mkdir models/
cd models
# resnet101 COCO trained model
wget https://www.dropbox.com/s/dpq6qv0efspelr3/faster_rcnn_1_10_9771.pth?dl=0
# resnet18 COCO trained model
wget TBD
cd ..
```

## Download dataset
We release two survillance videos you can test on. (coral.mp4 is from noscope)
A domain specific model is trained on such domain.

```
mkdir images
# download coral video
wget https://storage.googleapis.com/noscope-data/videos/coral-reef-long.mp4
mv coral-reef-long.mp4 coral.mp4

# download jackson video
wget TBD

```

## Setup dataset
If the models and the video are set, we can prepare the dataset.

1. The video is converted to JPEG.
2. Res101 model generates the teacher labels.
3. The dataset is prepared in a PASCAL_VOC format for training.

This is done in a single script.

Just run:

```
python make_dataset.py
```

### shortcut..
We prepared a dataset.tar in the link bellow, if you want to take a short cut.

Download (TBD) and place it in data dir.

# Training Domain Specific Models!

# Evaluation!


