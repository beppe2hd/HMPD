import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import PIL
import cv2

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
from skimage.draw import rectangle_perimeter
from glob import glob
import pathlib
from pathlib import Path
import uuid

class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.data = []

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max
        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:]+self.data[:self.cur]

    def append(self,x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

class BBMerge:
    def __init__(self, regionsBB):
        self.void = False
        if len(regionsBB)>0:
            self.regionsBB = regionsBB
            self.frame = self.regionsBB.frame.values[0]
        else:
            self.void = True

    # Returns true if two rectangles(l1, r1)
    # and (l2, r2) overlap
    def doOverlap(self, bb_a, bb_b):

        x1a = bb_a[0]
        x2a = bb_a[2]
        y1a = bb_a[1]
        y2a = bb_a[3]

        x1b = bb_b[0]
        x2b = bb_b[2]
        y1b = bb_b[1]
        y2b = bb_b[3]

        # To check if either rectangle is actually a line
        # For example : l1 ={-1,0} r1={1,1} l2={0,-1} r2={0,1}
        if (x1a == x2a or y1a == y2a or x1b == x2b or y1b == y2b):
            # the line cannot have positive overlap
            return False
        # If one rectangle is on left side of other
        if (x1a >= x2b or x1b >= x2a):
            return False
        # If one rectangle is above other
        if (y1a >= y2b or y1b >= y2a):
            return False
        return True

    def mysearch(self, currentid, indexes, currentlabel):
        bb_a = self.regionsBB.loc[currentid].bb
        foundIntersection = 0
        for j in indexes:
            if self.regionsBB.loc[j].label == -1:
                bb_b = self.regionsBB.loc[j].bb
                if self.doOverlap(bb_a, bb_b):
                    foundIntersection = 1
                    self.regionsBB.at[currentid, 'label'] = currentlabel
                    self.regionsBB.at[j, 'label'] = currentlabel
                    indexesB = indexes[:]
                    indexesB.remove(j)
                    self.mysearch(j, indexesB, currentlabel)
        if foundIntersection == 0:
            self.regionsBB.at[currentid, 'label'] = currentlabel

    def merge(self):
        if self.void:
            return []
        indexes = self.regionsBB.index.tolist()
        currentlabel = 0
        while len(indexes) > 0:
            currentid = indexes[0]
            indexes.remove(currentid)  # = indexes[1:]
            if self.regionsBB.loc[currentid].label == -1:
                self.mysearch(currentid, indexes, currentlabel)
                currentlabel += 1

        boundinboxInFrame = []
        labels = list(set(self.regionsBB.label.tolist()))
        for currentLabel in labels:
            bbsToMarge = self.regionsBB[self.regionsBB.label == currentLabel]
            arr = np.array(bbsToMarge.bb.to_list())
            mtx = np.asmatrix(arr)
            outbb = [
                mtx[:, 0].min(),
                mtx[:, 1].min(),
                mtx[:, 2].max(),
                mtx[:, 3].max()
            ]
            boundinboxInFrame.append({'frame': self.frame, 'bb': outbb})#(pd.DataFrame((self.frame, outbb)))

        #boundinboxInFrame = pd.concat(boundinboxInFrame)
        #boundinboxInFrame.rename(columns={0: 'frame', 1: 'bb'}, inplace=True)
        return pd.DataFrame(boundinboxInFrame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model file')
    parser.add_argument('--video', type=str, help='video to test')
    opt = parser.parse_args()
    modelPath = opt.model
    video = opt.video
    transform_normalize_mean = (0.5, 0.5, 0.5)
    transform_normalize_var = (0.5, 0.5, 0.5)
    x_nmp = RingBuffer(30)
    x_mp = RingBuffer(30)

    transform = transforms.Compose(
        [
            transforms.Resize((80,80)),
            transforms.ToTensor(),
            transforms.Normalize(transform_normalize_mean, transform_normalize_var),
        ]
    )

    device = 'cpu'
    model = torch.jit.load(modelPath)
    model.eval()
    model.to(device)

    thresholdSize = 150
    historyMOG = 100
    varThresholdMOG = 6
    morphKernel = 5
    mergeBox = 1
    maxNumOfRegions = 300

    #start video analysys
    thresholdSize = 150
    historyMOG = 100
    varThresholdMOG = 6
    morphKernel = 5
    mergeBox = 1
    maxNumOfRegions = 300
    startFrame = 40
    morphKernel = int(morphKernel)
    thresholdSize = thresholdSize
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphKernel, morphKernel))
    fgbgAdaptiveGaussain = cv2.createBackgroundSubtractorMOG2(history=historyMOG,
                                                              varThreshold=varThresholdMOG)

    cap = cv2.VideoCapture(video)

    num_of_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out = cv2.VideoWriter(f'./out.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
    #num_of_frame = 1500
    for x in range(1, num_of_frame):
        ret, frame = cap.read()
        if x > startFrame:
            fgbgAdaptiveGaussainmask = fgbgAdaptiveGaussain.apply(frame)
            fgbgAdaptiveGaussainmask = cv2.morphologyEx(fgbgAdaptiveGaussainmask, cv2.MORPH_CLOSE, kernel)

            label_im = label(fgbgAdaptiveGaussainmask)
            regions = regionprops(label_im)
            regions = [r for r in regions if r.area > thresholdSize]
            # check if the number of detected regions excede a given threshold.
            # This should avoid anomalus  detection due to repentine background changes
            print(f"in {x} there are {len(regions)} regions")
            if len(regions) < maxNumOfRegions:
                regionsBB = [(x, r.bbox, -1) for r in regions]
                regionsBB_DF = pd.DataFrame(regionsBB)
                regionsBB_DF.rename(columns={0: 'frame', 1: 'bb', 2: 'label'}, inplace=True)
                regionsBBFiltered = regionsBB_DF
                bbMerge = BBMerge(regionsBB_DF)
                regionsBBFiltered = bbMerge.merge()
                if len(regionsBBFiltered) > 0:
                    regionsBB = [(x, bb) for bb in regionsBBFiltered.bb]

                classValue = []
                for i, region in enumerate(regionsBB):
                    bb = region[1]
                    roi_P = frame[bb[0]:bb[2], bb[1]:bb[3]]
                    img = PIL.Image.fromarray(roi_P)
                    img = img.convert("RGB")
                    img = transform(img)
                    img = img.unsqueeze(0)
                    img.to(device)
                    model.eval()
                    prediction = model(img)
                    prediction = prediction.argmax()
                    classValue.append(prediction)
                    print(f"Element {i} in frame {region[0]} belongs to class {prediction}")

                for i, region in enumerate(regionsBB):
                    bb = region[1]
                    rr, cc = rectangle_perimeter(bb[0:2], end=bb[2:4], shape=frame.shape)
                    if classValue[i] == 0:
                        frame[rr, cc] = [255, 0, 0]
                    else:
                        frame[rr, cc] = [0, 255, 0]

                x_mp.append(classValue.count(1))
                x_nmp.append(classValue.count(0))

            mp_mean = np.array(x_mp.get()).mean()

            infoIMG = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(infoIMG, f"Frame:{x}",(30, 20) , 2, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(infoIMG, f"MP density is {mp_mean:.2f} particles/ml", (30, 50), 2, 0.7,
                        (0, 255, 0), 2)
            x_offset = 40
            y_offset = 40
            frame[y_offset:y_offset + infoIMG.shape[0], x_offset:x_offset + infoIMG.shape[1]] = infoIMG

            out.write(frame)

    out.release()