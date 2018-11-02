#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 08:42:47 2018

@author: kentaroy47
"""

import numpy as np
import copy
import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
parser.add_argument('--dataset', dest='dataset',
                      help='training dataset', type=str)
args = parser.parse_args()
target=args.dataset

xmlsource = '/home/ken/datasets/VOC2007/VOCdevkit/VOC2007/Annotations/'
datasource = '/data2/lost+found/img/'+target+'_train/*'
resultsdir = 'output/baseline/'+target+'train-res101.pkl'
valdir = 'output/baseline/'+target+'val-res101.pkl'
targetdir = 'output/'+target+'-train-labels-res101/'
if not os.path.isdir(targetdir):
    subprocess.call("mkdir "+targetdir, shell=True)
    
trainfile = 'trainval_'+target+'.txt'

train_num = 30000
THRESH = 0.5
OBJECT_ONLY = False
coco=False

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

pascal_classes = np.asarray(['__background__',
                           'aeroplane', 'bicycle', 'bear', 'boat',
                           'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse',
                           'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'truck'])

coco_classes = np.asarray(['__background__',"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
                               "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
                               "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie",
                               "suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
                               "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass",
                               "cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
                               "carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
                               "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
                               "microwave","oven","toaster","sink",
                               "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"])


            
# get xml files
import glob
import pickle
import xmltodict
import os

#with open(datasource) as f:
files = sorted(glob.glob(datasource))

trainvals = []
outlists = []
MAX_SAMPLE = 10000
for file in files:
     trainvals.append(os.path.basename(file)[:-4])


# make template
with open('images/000001.xml') as fd:
    doc = xmltodict.parse(fd.read())
del doc['annotation']['size']
del doc['annotation']['filename']
template = copy.deepcopy(doc['annotation']['object'])
template_all = copy.deepcopy(doc)


results2 = pickle.load(open(resultsdir, "rb"))
results=[]
if len(results2)>21:
    for n in pascal_classes:
        for i,cls in enumerate(coco_classes):
            if n == cls:
                print(cls)
                print(i)
                results.append(results2[i])
else:
    results=results2
    
            
vals2 = pickle.load(open(valdir, "rb"))
vals=[]
if len(vals2)==81:
    for n in pascal_classes:
        for i,cls in enumerate(coco_classes):
            if n == cls:
                print(cls)
                print(i)
                vals.append(vals2[i])
                
    with open(valdir, 'wb') as f:
          pickle.dump(vals, f, pickle.HIGHEST_PROTOCOL)

      
classes=pascal_classes
        
box = []
boxsize = []
nbox = []
counter = np.zeros(len(classes))
#trainclass = classes[2],classes[7],classes[14],classes[15]
trainclass = classes
train_num=len(results[0])

for i,file in enumerate(trainvals[1:train_num]):
    
    del doc['annotation']['object'] 
    doc['annotation']['object'] = copy.deepcopy(template)
        
    try:
        while len(doc['annotation']['object'])>0:
            temp = doc['annotation']['object'].pop()
    except:
        None
        
    flag = 0
    
    
    for ncls, cls in enumerate(classes):
        bboxes=[]
        result = results[ncls][i]
        for out in result:
            if out[4] > THRESH and cls in trainclass: # confident
                a = temp
                a['bndbox']['xmin'] = int(np.floor(out[0]))
                a['bndbox']['ymin'] = int(np.floor(out[1]))
                a['bndbox']['xmax'] = int(np.floor(out[2]))
                a['bndbox']['ymax'] = int(np.floor(out[3]))
                a['name'] = cls
                
#                print("file",file,a)
                # filter small bbox
                if min(out[2]-out[0],out[3]-out[1]) > 20:                    
                    if min(out[2]-out[0],out[3]-out[1]) < 40:
                        a['difficult'] = 1
                    doc['annotation']['object'].append(copy.deepcopy(a))
                    flag += 1
                    counter[ncls] +=1
                    bboxes.append(out[0:4])
                
                # monitor bbox
                
                box.append(copy.deepcopy(a))
                bsize = min(out[2]-out[0],out[3]-out[1])
                boxsize.append(bsize)
                
        #check overlap of boxes        
#        for numbox, bbox in enumerate(bboxes):
#            iou=0
#            for ntbox, testbbox in enumerate(bboxes):
#                if numbox!=ntbox:
#                    iout = bb_intersection_over_union(bbox, testbbox)
#                    if iout>iou:
#                        iou=iout
#            if iou>0.5:
#                print(iou)
#                doc['annotation']['object'][numbox]['difficult']=1
                        
    nbox.append(flag)
    if flag == 0:
        print("no target was added!")
        if not OBJECT_ONLY:
            outlists.append(file)
            write = targetdir + file + '.xml'
#            with open(write, "w") as f:
#                f.write(xmltodict.unparse(doc, pretty=True))
                
    else:
        outlists.append(file)
        write = targetdir + file + '.xml'
#        print("writing")
        with open(write, "w") as f:
            f.write(xmltodict.unparse(doc, pretty=True))
    #print(xmltodict.unparse(doc, pretty=True))
    

with open(trainfile, "w") as f: 
    for trainval in outlists:
        f.write(trainval + '\n') 

import subprocess

command = "cp "+trainfile+" data/VOCdevkit"+target+"/VOC2007/ImageSets/Main/"
subprocess.call(command, shell=True)

#files = sorted(glob.glob(datasource2))
#
#trainvals = []
#outlists = []
#MAX_SAMPLE = 10000
#for file in files:
#     trainvals.append(os.path.basename(file)[:-4])


## write val
#outlists = []
#
#for i,file in enumerate(trainvals):
#    
#    del doc['annotation']['object'] 
#    doc['annotation']['object'] = copy.deepcopy(template)
#        
#    try:
#        while len(doc['annotation']['object'])>0:
#            temp = doc['annotation']['object'].pop()
#    except:
#        None
#        
#    flag = 0
#    
#    
#    for ncls, cls in enumerate(classes):
#        result = results[ncls][i]
#        for out in result:
#            if out[4] > THRESH and cls in trainclass: # confident
#                a = temp
#                a['bndbox']['xmin'] = int(np.floor(out[0]))
#                a['bndbox']['ymin'] = int(np.floor(out[1]))
#                a['bndbox']['xmax'] = int(np.floor(out[2]))
#                a['bndbox']['ymax'] = int(np.floor(out[3]))
#                a['name'] = cls
#                print("file",file,a)
#                # filter small bbox
#                if min(out[2]-out[0],out[3]-out[1]) > 10 and flag < 20 and counter[ncls] < MAX_SAMPLE:
#                    doc['annotation']['object'].append(copy.deepcopy(a))
#                    flag += 1
#                    counter[ncls] +=1
#                # monitor bbox
#
#                box.append(copy.deepcopy(a))
#                bsize = min(out[2]-out[0],out[3]-out[1])
#                boxsize.append(bsize)
#                
#                
#    nbox.append(flag)
#    if flag == 0:
#        print("no target was added!")
#        if not OBJECT_ONLY:
#            outlists.append(file)
#            write = targetdir + file + '.xml'
#            with open(write, "w") as f:
#                f.write(xmltodict.unparse(doc, pretty=True))
#                
#    else:
#        outlists.append(file)
#        write = targetdir + file + '.xml'
#        with open(write, "w") as f:
#            f.write(xmltodict.unparse(doc, pretty=True))
#    #print(xmltodict.unparse(doc, pretty=True))
#    
#
#with open(valfile, "w") as f: 
#    for trainval in outlists:
#        f.write(trainval + '\n') 
#    
