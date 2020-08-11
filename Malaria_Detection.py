import cv2,os
import numpy as np
import csv
import glob

label = "Parasitized"
dirList = glob.glob("cell_images/"+label+"/*.png")
file = open("csv/dataset.csv", "a")

# iterate through files in dataset folder
for img_path in dirList:
    im = cv2.imread("img_path")
    # smoothen image and reduce detail using gaussian blur
    im = cv2.GaussianBlur(im,(5,5),2)
    # convert to gray scale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(im_gray, 127,255,0)
    # store contours of images
    contours,_=cv2.findContours(thresh,1,2)

    file.write(label)
    file.write(",")

    # store data into csv file
    for i in range(5):
        try:
            area = cv2.contourArea(contours[i])
            file.write(str(area))
        except:
            file.write("0")
        file.write(",")

    file.write("\n")