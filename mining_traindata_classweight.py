# -*- coding: utf-8 -*-

import numpy as np
import argparse

def get_results(boxes, n):
    out=[]
    for box in boxes:
        out.append(box[n])
    return out

def flatten(libs):
    flatlist = []
    confidence = []
    idx = []
    for i,minilib in enumerate(libs):
#        for minilib in lib:
            flatlist.append(minilib[0:4])
#            print(minilib)
            confidence.append(minilib[4])
            idx.append(i)
    return flatlist, confidence, idx
def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap

# simple Bbox loss
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')

  parser.add_argument('--truth', dest='truth',
                      help='directory to load models', default='output/baseline/taipei2train-res101.pkl')
  parser.add_argument('--result', dest='result',
                      help='directory to load models', default='output/pascal_voc_taipei2-res18.pkl')
  parser.add_argument('--coco', dest='coco',
                      help='directory to load models', default=False, type=bool)
  parser.add_argument('--save', dest='save',
                      help='directory to load models', default=False, type=bool)
  parser.add_argument('--box', dest='box',
                      help='directory to load models', default='output/bbox-pascal_voc_taipei2-res18.pkl')
  parser.add_argument('--truthbox', dest='truthbox',
                      help='directory to load models', default='output/baseline/bbox-taipei2_test-res101.pkl')
  parser.add_argument('--nout', dest='nout',
                      help='directory to load models', default=100, type=int)
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset', type=str)

  args = parser.parse_args()
  return args

if __name__ == '__main__':

  args = parse_args()
  target=args.dataset
  trainclass=['bus', 'car', 'motorbike', 'person' ]

  pascal_classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])
    
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
  import pickle       
  
  truth_boxes2 = pickle.load(open(args.truth, "rb"))
  truth_boxes=[]
  if len(truth_boxes2)==81:      
      for n in pascal_classes:
            for i,cls in enumerate(coco_classes):
                if n == cls:
#                    print(cls)
#                    print(i)
                    truth_boxes.append(truth_boxes2[i])
  else:
      truth_boxes=truth_boxes2
      
  all_boxes2 = pickle.load(open("output/pascal_voc_"+target+"-res18.pkl", "rb"))
  small_bboxes = pickle.load(open("output/bbox-pascal_voc_"+target+"-res18.pkl", "rb"))
  big_bboxes = pickle.load(open("output/baseline/bbox-"+target+"_test-res101.pkl", "rb"))
  
  box_losses = []
  im = np.zeros([1000, 562])
  small_bbox = np.round(small_bboxes[0])
  big_bbox = np.round(big_bboxes[0])
  
  im_small = im
  ovthresh = 0.5
  
  TRUTH_THRESHOLD=0.75
  diflist=[]
  counter=np.zeros(21)
  for i in range(len(truth_boxes[:])):
      diflist2=[]
      for ii in range(len(truth_boxes[i][:])):          
          
          poplist = []
          dif=0
          for iii in range(len(truth_boxes[i][ii][:])):
              
              out = truth_boxes[i][ii][iii][0:4]
              
              if truth_boxes[i][ii][iii][4] < TRUTH_THRESHOLD:
                  poplist.append(iii)
              elif min(out[2]-out[0],out[3]-out[1]) < 25:
                  dif+=1
              else:
                  counter[i]+=1
          diflist2.append(dif)
          if len(poplist) is not 0:
              for pop in reversed(poplist):
    #                  truth_boxes[i][ii] = np.delete(truth_boxes[i][ii], pop, axis=0)
                  truth_boxes[i][ii] = np.delete(truth_boxes[i][ii], pop, axis=0)  
      diflist.append(diflist2)
  
#  for bbox in small_bbox[0,:100,:]:
#      for x in range(int(bbox[0]), int(bbox[2])+1):
#          for y in range(int(bbox[1]), int(bbox[3])+1):
#              im_small[x,y]=1
  
  truth_boxes2 = truth_boxes  
  loss_list=[]
  for img_num in range(len(big_bboxes)):   
      # filter truth by confidence > 50%
      # sample difficult
      all_boxes = get_results(all_boxes2, img_num)
      truth_boxes = get_results(truth_boxes2, img_num)
      if img_num%100==0:
          print(img_num)
      aps = []
      for ncls, cls in enumerate(pascal_classes):
          if cls is not "__background__":
    #      if cls is "car":
              print("detection for ", cls)
              BBs = all_boxes[ncls][:]
              BBGTs = truth_boxes[ncls][:]
              # flatten
              BB, confidence, idx = flatten(BBs)
              confidence = np.asarray(confidence)
              BB = np.asarray(BB)
              idx = np.asarray(idx)
              
              BBGT_flat, gtconf, gtidx = flatten(BBGTs)
              gtidx = np.asarray(gtidx)
              BBGT_flat = np.asarray(BBGT_flat)
              
              nd = len(BBGT_flat)
    #          nd = len(BB)
              tp = np.zeros(nd)
              fp = np.zeros(nd)
              npos = nd
              
              det=[False] * len(BBGTs)
    
              
              if len(BB) > 0:
                    # sort by confidence
                    sorted_ind = np.argsort(-confidence)
                    sorted_scores = np.sort(-confidence)
                    BB = BB[sorted_ind, :]
                    idx = [idx[x] for x in sorted_ind]
    
                    
                    # go down dets and mark TPs and FPs
                    for d in range(nd):
                      if d >= len(BB):
                          continue
                      bb = BB[d,:].astype(float)
                      ovmax = -np.inf
                      index = -np.inf
    
                      try:
                          BBGT = BBGTs[idx[d]]
                      except:
                          BBGT = []
                
                      if len(BBGT) > 0:
                          
                        # compute overlaps
                        # intersection
                        try:
                            ixmin = np.maximum(BBGT[:,0], bb[0])
                        except:
                            BBGT=BBGT.reshape(1,5)
                            ixmin = np.maximum(BBGT[:,0], bb[0])
                        iymin = np.maximum(BBGT[:,1], bb[1])
                        ixmax = np.minimum(BBGT[:,2], bb[2])
                        iymax = np.minimum(BBGT[:,3], bb[3])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih
                
                        # union
                        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                               (BBGT[:,2] - BBGT[:,0] + 1.) *
                               (BBGT[:,3] - BBGT[:,1] + 1.) - inters)
                
                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)
                
                      # penaltize by confidence?? not much change
                      if ovmax > ovthresh:
                          tp[d] = 1.
                      else:
                          fp[d] = 1.
    #                  if ovmax > ovthresh:
    #                        if not det[jmax]:
    #                            tp[d] = 1.
    #                            det[jmax] = 1
    #                        else:
    #                            fp[d] = 1.
    #                  else:
    #                        fp[d] = 1.
    
            
              # compute precision recall
              fp = np.cumsum(fp)
              tp = np.cumsum(tp)
              rec = tp / float(npos)
              # avoid divide by zero in case the first detection matches a difficult
              # ground truth
              prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
              ap = voc_ap(rec, prec, use_07_metric=True)
              print("rec:", rec)
              try:
                  # normalize by number of samples
                  if npos!=0:
                      print("tp",tp)
                      print("npos",npos)
                      eps=0
                      if sum(tp)==0:
                          eps=0.5
                      rec_loss = np.abs(float(npos)/(sum(tp)+eps))#/counter[ncls]
                      print("loss",rec_loss)
                  else:
                      rec_loss=0
              except:
                  rec_loss = 0
              aps.append(rec_loss)
      loss_list.append(aps)
  
  
  loss_rank = []
  for ncls, cls in enumerate(pascal_classes):
      loss_total = []  
      for n, loss in enumerate(loss_list):
          if cls in trainclass:
              loss_total.append(loss_list[n][ncls])
      loss_rank.append(np.argsort(np.array(loss_total)))

  out_list = []      
  nout = args.nout
  length=len(loss_list)
  
  target2 = length - np.round(nout/len(trainclass))
    
  for ncls, cls in enumerate(pascal_classes):
      if cls in trainclass:
          for n, loss in enumerate(loss_rank[ncls]):
              if loss>=target2:
                  out_list.append(n)
  
import glob    
import os 
trainfile = 'trainval_mine_'+target+'.txt'
datasource = '/data2/lost+found/img/'+target+'_train/*'

#with open(datasource) as f:
files = sorted(glob.glob(datasource))

trainvals = []
outlists = []
MAX_SAMPLE = 10000
for file in files:
     trainvals.append(os.path.basename(file)[:-4])  

with open(trainfile, "w") as f: 
    for n, trainval in enumerate(trainvals):
        if n in out_list:
            f.write(trainval + '\n')           
            
counter2 = np.zeros(21)
truth_boxes = truth_boxes2
for i in range(len(truth_boxes[:])):
      for ii in range(len(truth_boxes[i][:])):          
          if ii in out_list:
              for iii in range(len(truth_boxes[i][ii][:])):
                  
                  if truth_boxes[i][ii][iii][4] > TRUTH_THRESHOLD:
                      counter2[i]+=1     

out_list2=[]
countsum=0
for ncls, cls in enumerate(pascal_classes):
      if cls in trainclass:
          countsum+=counter2[ncls]

for ncls, cls in enumerate(pascal_classes):
      if cls in trainclass:
          target2 = length - np.round(nout/countsum*counter2[ncls])
          print(target2)
          for n, loss in enumerate(loss_rank[ncls]):
              if loss>=target2:
                  out_list2.append(n)
                  
counter3 = np.zeros(21)
truth_boxes = truth_boxes2
for i in range(len(truth_boxes[:])):
      for ii in range(len(truth_boxes[i][:])):          
          if ii in out_list2:
              for iii in range(len(truth_boxes[i][ii][:])):                  
                  if truth_boxes[i][ii][iii][4] > TRUTH_THRESHOLD:
                      counter3[i]+=1       