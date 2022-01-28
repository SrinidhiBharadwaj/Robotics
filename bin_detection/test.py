'''
File created to test out autograder results
Not part of the project
'''

from bin_detector import BinDetector
import cv2, os, glob
import numpy as np 


class TestBin():
  def __init__(self):
    self.detector = BinDetector()
    #self.classifier = PixelClassifier()
    self.iou_thre = 0.5  # threshold for bounding box IOU

  def iou(self, box1, box2):
    '''
    Computes the intersection over union of two bounding boxes box = [x1,y1,x2,y2]
    where (x1, y1) and (x2, y2) are the top left and bottom right coordinates respectively
    '''
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, (x2 - x1 + 1)) * max(0, (y2 - y1 + 1))
    union_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) + (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1) - inter_area
    return inter_area/union_area

  def compare_boxes(self, true_boxes, estm_boxes):
    '''
    Compares the intersection over union of two bounding box lists.
    The iou is computed for every box in true_boxes sequentially with respect to each box in estm_boxes.
    If any estm_box achieves an iou of more than 0.5, then the true box is considered accurately detected.
    '''
    num_true_boxes = len(true_boxes)
    if num_true_boxes == 0:
      return float(len(estm_boxes) == 0)

    accuracy = 0.0
    for box1 in true_boxes:
      for box2 in estm_boxes:
        if self.iou(box1,box2) >= self.iou_thre:
          accuracy += 1.0
          break
    return accuracy / num_true_boxes


  def test_partial_boxes(self, set_score=None):
    """Bin Detection"""
    img_folder = "data/validation/"
    img_list = glob.glob(img_folder+"/*.jpg")
    gt_box_list = glob.glob(img_folder+"/*.txt")
    img_list.sort()
    gt_box_list.sort()
    score = 0
    print(img_list)
    for i in range(len(img_list)):
        #print("Scoring started")
        img = cv2.imread(img_list[i])
        l=[]
        with open(gt_box_list[i], 'rU') as f:
            for e1 in f:
                line = e1.split(',')
            for e2 in line:
                e2 = e2.replace('[[', '')
                e2 = e2.replace(']]\n', '')
                e2 = e2.replace(']', '')
                e2 = e2.replace('[', '')
                e2 = e2.replace('\n', '')
                if len(e2) > 0:
                    l.append(int(e2))
        gt_boxes = []
        for j in range(int(len(l)/4)):
          gt_boxes.append(l[j*4:j*4 + 4]) # ground truth bottom left and top right coordinates

        mask_img = self.detector.segment_image(img)

        pred_boxes = self.detector.get_bounding_boxes(mask_img) # predicted coordinates
        print(gt_boxes)
        accuracy = self.compare_boxes(gt_boxes, pred_boxes)
        score+=accuracy
        print("{} for file {}".format(accuracy, img_list[i]))
    return score

if __name__ == '__main__':
  test_bin = TestBin()
  score = test_bin.test_partial_boxes()
  print(score)
  

