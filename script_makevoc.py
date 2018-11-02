# -*- coding: utf-8 -*-

"""
script for setting up pascal-like datasets
copyright:kentaroy47
10/2/2018
"""

import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
parser.add_argument('--dataset', dest='dataset',
                      help='training dataset', type=str)
args = parser.parse_args()
target=args.dataset

# make dir in datasets
datasetname = "VOCdevkit"+target
command="mkdir data/"+datasetname
subprocess.call(command, shell=True)
command="mkdir data/"+datasetname+"/VOC2007/"
subprocess.call(command, shell=True)
command="mkdir data/"+datasetname+"/VOC2007/JPEGImages"
subprocess.call(command, shell=True)
command="mkdir data/"+datasetname+"/VOC2007/Annotations"
subprocess.call(command, shell=True)
command="mkdir data/"+datasetname+"/VOC2007/ImageSets"
subprocess.call(command, shell=True)
command="mkdir data/"+datasetname+"/VOC2007/ImageSets/Main"
subprocess.call(command, shell=True)

# generate xml
command="python xml_makelabels_domain.py --dataset "+target
subprocess.call(command, shell=True)

# make link
command="cp output/"+target+"-train-labels-res101/"+target+"_train* data/"+datasetname+"/VOC2007/Annotations"
subprocess.call(command, shell=True)
command="cp images/"+target+"_train/* data/"+datasetname+"/VOC2007/JPEGImages/"
subprocess.call(command, shell=True)

# copy text file to Main
command="cp trainval_"+target+".txt data/"+datasetname+"/VOC2007/ImageSets/Main/"
subprocess.call(command, shell=True)

# make models and copy
command="mkdir models/res18/pascal_voc_"+target
subprocess.call(command, shell=True)
command="cp models/faster_rcnn_500_40_625.pth models/res18/pascal_voc_"+target
subprocess.call(command, shell=True)
#command="mkdir models/squeeze/pascal_voc_"+target
#subprocess.call(command, shell=True)
#command="cp models/squeeze/faster_rcnn_500_40_625.pth models/squeeze/pascal_voc_"+target
#subprocess.call(command, shell=True)