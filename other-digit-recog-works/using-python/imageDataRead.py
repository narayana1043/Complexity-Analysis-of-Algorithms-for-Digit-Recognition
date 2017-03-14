
from PIL import Image
import numpy as np
import csv
import pickle

trainDataLoc = 'E:/google_drive/project_works/digit-recognition/data/train.csv'
testDataLoc = 'E:/google_drive/project_works/digit-recognition/data/test.csv'

def displayImage(imgDataAr):

    img = Image.new('P', (28, 28))  # create a new black image

    pixels = img.load() # create the pixel map

    for newRow, Row in zip(range(img.size[0]), imgDataAr):  # for every pixel:
        for newPix, Pix in zip(range(img.size[1]), Row):
            pixels[newRow, newPix] = (int(Pix))  # set the colour accordingly

    img.show()

def threshold(imageArray):

    newAr = imageArray
    newAr.setflags(write= True)

    for row in range(newAr.shape[0]):
        for pix in range(newAr.shape[1]):
            if int(newAr[row,pix]) > 0:
                newAr[row, pix] = 255

            else:
                newAr[row, pix] = 0

    #print(newAr)

    return newAr

def imageDataReader(trainData):

    for eachImg in trainData:
        # print(eachImg)
        classLabel = eachImg[0]
        imgData = np.asarray(eachImg[1:]).reshape((28, 28)).astype(np.int16)
        print(type(imgData))
        # displayImage(imgData)
        imgDataAfterThershold = threshold(imgData)
        # displayImage(imgDataAfterThershold)
        # print(imgDataAfterThershold)
        print(np.subtract(imgData, imgDataAfterThershold))

        break

def readTrainData():

    trainDataFile = open(trainDataLoc, 'r')
    trainDataReader = csv.reader(trainDataFile)
    trainData = list(trainDataReader)
    trainData = trainData[1:]

    #imageDataReader(trainData)

    dataDict = {}

    for num in range(10):

        dataDict[num] = list()

    for eachImg in trainData:

        npImageMat = np.asarray(eachImg[1:]).reshape(28,28).astype(np.bool)
        # print(npImageMat)
        dataDict[int(eachImg[0])].append(npImageMat)

    with open('./pickles/trainDataDict.p','wb') as handle:
        pickle.dump(dataDict, handle)

def readTestData():

    testDataFile = open(testDataLoc, 'r')
    testDataReader = csv.reader(testDataFile)
    testData = list(testDataReader)
    testList = list()

    for eachImg in testData[1:]:

        npImageMat = np.asarray(eachImg).reshape(28,28).astype(np.bool)
        testList.append(npImageMat)

    with open('./pickles/testList.p', 'wb') as handle:
        pickle.dump(testList, handle)

#readTrainData()
readTestData()