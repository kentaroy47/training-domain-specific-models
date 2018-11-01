# training-domain-specific-models
Training domain specific object detection models.

Faster-RCNN implementation is based on faster-rcnn.pytorch by jwyang. Thanks!
https://github.com/jwyang/faster-rcnn.pytorch

# Preparation

# clone repo
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
```

## download dataset
we prepaired 
```
wget coral...
```

## setup dataset
python ..