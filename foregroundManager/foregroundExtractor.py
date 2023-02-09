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


class ForegraoundExtractor:

    def __init__(self, path = '', thresholdSize = 500,
                 historyMOG = 200, varThresholdMOG = 16, morphKernel = 3, mergeBox = True, maxNumOfRegions=50, startFrame = 1):
        self.path = path
        self.cap = cv2.VideoCapture(f'{path}/P.avi')
        self.varThresholdMOG = varThresholdMOG
        self.historyMOG = historyMOG
        self.fgbgAdaptiveGaussain = cv2.createBackgroundSubtractorMOG2(history=self.historyMOG, varThreshold=self.varThresholdMOG)
        self.experimentName = ''
        self.experimentDir = ''
        self.morphKernel = int(morphKernel)
        self.thresholdSize = thresholdSize
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morphKernel, self.morphKernel))
        self.mergeBox = mergeBox
        self.maxNumOfRegions = maxNumOfRegions
        self.startFrame = startFrame



    def processFile(self):

        boundinboxInFrame = []
        num_of_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.experimentName = f'th_{self.thresholdSize}_h{self.historyMOG}_var{self.varThresholdMOG}_mk{self.morphKernel}_merge{self.mergeBox}_maxRegion{self.maxNumOfRegions}_startFrame{self.startFrame}/'
        self.experimentDir = f'{self.path}/{self.experimentName}'
        Path(self.experimentDir).mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(f'{self.experimentDir}/out.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))

        for x in range(1, num_of_frame):
            ret, frame = self.cap.read()
            if x > self.startFrame:
                fgbgAdaptiveGaussainmask = self.fgbgAdaptiveGaussain.apply(frame)
                fgbgAdaptiveGaussainmask = cv2.morphologyEx(fgbgAdaptiveGaussainmask, cv2.MORPH_CLOSE, self.kernel)

                label_im = label(fgbgAdaptiveGaussainmask)
                regions = regionprops(label_im)
                regions = [r for r in regions if r.area > self.thresholdSize]
                # check if the number of detected regions excede a given threshold.
                # This should avoid anomalus  detection due to repentine background changes
                print(f"in {x} there are {len(regions)} regions")
                if len(regions)<self.maxNumOfRegions:
                    print(f'{x} of {num_of_frame} with {len(regions)} regions')
                    regionsBB = [(x, r.bbox, -1) for r in regions]
                    regionsBB_DF = pd.DataFrame(regionsBB)
                    regionsBB_DF.rename(columns={0: 'frame', 1: 'bb', 2: 'label'}, inplace=True)
                    regionsBBFiltered = regionsBB_DF
                    if self.mergeBox == True:
                        bbMerge = BBMerge(regionsBB_DF)
                        regionsBBFiltered = bbMerge.merge()
                    if len(regionsBBFiltered)>0:
                        regionsBB = [(x, bb) for bb in regionsBBFiltered.bb]
                    boundinboxInFrame.append(pd.DataFrame(regionsBBFiltered))

                    for region in regionsBB:
                        bb = region[1]
                        rr, cc = rectangle_perimeter(bb[0:2], end=bb[2:4], shape=frame.shape)
                        frame[rr, cc] = [255, 0, 0]

                cv2.putText(frame, str(x), (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0))
                out.write(frame)

        out.release()

        boundinboxInFrame = pd.concat(boundinboxInFrame)
        boundinboxInFrame.rename(columns={0: 'frame', 1: 'bb'}, inplace=True)
        boundinboxInFrame.to_parquet(f'{self.experimentDir}/out.parquet', index=False)

        self.cap.release()
        cv2.destroyAllWindows()

        return self.experimentName
