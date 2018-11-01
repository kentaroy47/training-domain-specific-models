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
command="cp -rf /home/ken/datasets/VOC2007/jackson/ /home/ken/datasets/VOC2007/"+target
subprocess.call(command, shell=True)

# make xml
command="python xml_makelabels_domain.py --dataset "+target
subprocess.call(command, shell=True)

# make symb link
command="rm /home/ken/datasets/VOC2007/"+target+"/VOC2007/Annotations"
subprocess.call(command, shell=True)
command="rm /home/ken/datasets/VOC2007/"+target+"/VOC2007/JPEGImages"
subprocess.call(command, shell=True)

command="ln -s /home/ken/distil/output/"+target+"-train-labels-res101/ /home/ken/datasets/VOC2007/"+target+"/VOC2007/Annotations"
subprocess.call(command, shell=True)
command="ln -s /data2/lost+found/img/"+target+"_train/ /home/ken/datasets/VOC2007/"+target+"/VOC2007/JPEGImages"
subprocess.call(command, shell=True)

# copy text file to Main
command="cp /home/ken/distil/trainval_"+target+".txt /home/ken/datasets/VOC2007/"+target+"/VOC2007/ImageSets/Main/"
subprocess.call(command, shell=True)

# make final link
command="ln -s /home/ken/datasets/VOC2007/"+target+" data/VOCdevkit"+target
subprocess.call(command, shell=True)

# make models and copy
command="mkdir models/res18/pascal_voc_"+target
subprocess.call(command, shell=True)
command="cp models/res18/faster_rcnn_500_40_625.pth models/res18/pascal_voc_"+target
subprocess.call(command, shell=True)
command="mkdir models/squeeze/pascal_voc_"+target
subprocess.call(command, shell=True)
command="cp models/squeeze/faster_rcnn_500_40_625.pth models/squeeze/pascal_voc_"+target
subprocess.call(command, shell=True)