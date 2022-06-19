from distutils.command.clean import clean
from unittest import result
from cv2 import GaussianBlur
import numpy as np
import cv2
import imutils
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage import io
import os
from matplotlib import image, pyplot as plt
from PIL import Image

#-------------------------------------------------------------------------------------------------------------------------------------------------->
#               PREPROCESSING , BORDER CLEANING , AREA CHECKING AND CHARACTER SEGMENTATION FUNCION OF TOP AND BOTTOM HALF OF LOCALIZED PALTE
#--------------------------------------------------------------------------------------------------------------------------------------------------->

#---------------preimage processing------>



def preProcessingImage(inpImage):

    grayFilter = cv2.cvtColor(inpImage,cv2.COLOR_BGR2GRAY)
    gaussianFilter = cv2.GaussianBlur(grayFilter,(5,5),0)
    thresholdFilter = cv2.threshold(gaussianFilter,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    morphFilter = cv2.morphologyEx(thresholdFilter,cv2.RETR_LIST,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    return morphFilter


#--------cleaning Borders ------------->

def cleanBorder(inpImg, radius):

    inpImgCopy = inpImg.copy()
    contours,hierarchy = cv2.findContours(inpImgCopy,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    imgRow = inpImgCopy.shape[0]
    imgCol = inpImgCopy.shape[1]

    contourList = []

    for index in np.arange(len(contours)):
        cnt = contours[indx]
        for points in cnt:
            rowContour = points[0][1]
            colContour = points[0][0]

            check1 = (rowContour >= 0 and rowContour < radius) or (rowContour >= imgRow - 1 - radius and rowContour < imgRow)
            check2 = (colContour >= 0 and colContour < radius) or (colContour >= imgCol - 1 - radius and colContour < imgCol)

            if check1 or check2:
                contourList.append(index)
                break

        for index in contourList:
            cv2.drawContours(inpImgCopy, contours, index, (0,0,0), -1)

        return inpImgCopy


#-------------------Area filter -------------->


def checkArea(inpImg, areaThres):

    inpImgCopy = inpImg.copy()
    contours,hierarchy = cv2.findContours(inpImgCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for index in contours:
        area = cv2.contourArea(index)
        shape = cv2.approxPolyDP(index, 0.04*cv2.arcLength(index, True), True)
        if area >= 0 and area <= areaThres and len(shape) == 4:
            cv2.drawContours(inpImgCopy, index, -1, (0,0,0), -1)
            break
    return inpImgCopy


def fillCharacters(inpImg):

    inpImgCopy = inpImg.copy()
    test = cv2.morphologyEx(inpImgCopy, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    return test

charactersTop = []
columnListTop = []
charactersBot = []
columnListBot = []

def characterSegment(inpImg, finalImg, mux):
    print("CHARACTER SEGMENT INITIALIZED...")
    print(".....................................")
    inpImgCopy = inpImg.copy()
    contours, hierarchy = cv2.findContours(inpImgCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    tempSum = 0
    tempCount = 0
    for cnts in contours:
        x,y,w,h = cv2.boundingRect(cnts)
        tempCount = tempCount + w
        print("No of boundedRECT CONTOUR::", tempCount)
        AvgWidth = tempSum + w
        tempSum = AvgWidth
        print("Width of CONTOUR,"+str(tempCount)+":",w)
        print("SUM::",AvgWidth)

    
    AverageWidth = tempSum / tempCount
    print("AVERAGE WIDTH OF RECT_CONTOURS", AverageWidth)
    i = 0
    for cnts in contours:
        x1,y1,w1,h1 = cv2.boundingRect(cnts)
        # compWidth1 = (2/3)*AverageWidth
        # compWidth2 = (1/3)*AverageWidth
        if w1 >= AverageWidth :
            i = i + 1
            print("Index of image name count::", i)
            cv2.rectangle(finalImg, (x1,y1), (x1+w1, y1+h1), (255, 255, 255), 0)
            roiImage = finalImg[y1:y1+h1, x1:x1+w1]
            # path = "/home/..."
            grayChar = cv2.cvtColor(roiImage, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("garsytaf",grayChar)
            #_, blackWhite = cv2.threshold(grayChar, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #_,invertImage = cv2.threshold(blackWhite,0,255,cv2.THRESH_BINARY_INV)
            #cv2.imshow("jklsjkl", invertImage)
            resizedImage = resize(grayChar, (30, 30))
            thresholdChar = resizedImage < threshold_otsu(resizedImage)

            print("_________________________________________________________________________")


            if mux == "top":
                charactersTop.append(thresholdChar)
                columnListTop.append(x1)
                cv2.imwrite(str(i)+"__"+"top"+".jpg",roiImage)
                # cv2.imshow(str(i) + "__" + str(w1) + "top" + ".jpg", roiImage)

            elif mux == "bottom":
                charactersBot.append(thresholdChar)
                columnListBot.append(x1)
                cv2.imwrite(str(i)+"__"+str(w1)+"bottom"+".jpg",roiImage)
                # cv2.imshow(str(i) + "__" + str(w1) + "bottom" + ".jpg", roiImage)


    return finalImg


#---------------------------------------------------------------------------------------------------------------------------------------------->
#                                        LICENSE PLATE LOCALIZATION FROM INPUT IMAGE
#---------------------------------------------------------------------------------------------------------------------------------------------->

# MAIN PROGRAM

# Read input image
i = 0
print("##LICENSE PLATE LOCALIZATION STARTED...")
inputImage = cv2.imread("test2.jpg")

inputImage = imutils.resize(inputImage, 500, 500, cv2.INTER_AREA)
cv2.imshow("Original Image", inputImage)
finalImage = imutils.resize(inputImage, 500, 500, cv2.INTER_AREA)
contourImage = imutils.resize(inputImage, 500, 500, cv2.INTER_AREA)


# convert image from BGR to HSV
hsvImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
# cv2.imshow("HSV image", hsvImage)


# Setting range for red color in hsv format 
lowRed1 = np.array([0, 50, 50])
upRed1 = np.array([10, 255, 255])
maskRed1 = cv2.inRange(hsvImage, lowRed1, upRed1)

lowRed2 = np.array([170, 50, 50])
upRed2 = np.array([180, 255, 255])
maskRed2 = cv2.inRange(hsvImage, lowRed2, upRed2)


# combining mask ranges into one image
mask = maskRed1 | maskRed2

resultImage = cv2.bitwise_and(inputImage, inputImage, mask=mask)
# cv2.imshow("HSV filtered image with red masking", resultImage)


# Applying filters 
grayImage = cv2.cvtColor(resultImage, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray image", grayImage)

gaussianBlur = cv2.GaussianBlur(grayImage, (5,5), 0)
# cv2.imshow("Gaussian blur", gaussianBlur)

# convert to binary 
ret, thresholdImage = cv2.threshold(gaussianBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("threshold", thresholdImage)

gradientImage = cv2.morphologyEx(thresholdImage,cv2.MORPH_GRADIENT, kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
# cv2.imshow("gradient image", gradientImage)


# Edge Detection 
cannyEdge = cv2.Canny(gradientImage, 100, 300)
cv2.imshow("final canny edge", cannyEdge)
print("preprocessing done...")


# Drawing contours 
contours,hierarchy = cv2.findContours(cannyEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

print("Drawing contours...")
imageIndex = 0
for cnts in contours:
    epsilon = 0.04*cv2.arcLength(cnts,True)
    approx = cv2.approxPolyDP(cnts, epsilon, True)
    cv2.drawContours(contourImage, cnts, -1, (0, 255, 0), 3)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(cnts)
        ar = float(w) / float(h)
        print("ASPECT RATIO OF BOUND_RECT::",ar)
        aspect = 4.7272
        minArea = 15*aspect*15
        maxArea = 125*aspect*125
        area = w*h

        if ar < 1:
            ar = 1/ar

        if ar >= 0.95 and ar <= 1.05:
            print("square")

        if (ar >= 1.33 and ar <= 2.0) and (area >= minArea and area <= maxArea):
            imageIndex += 1
            rectImage = cv2.rectangle(inputImage, (x,y), (x+w, y+h), (0, 255, 0), 2)

            # selecting region of interest of the image 
            roiImage = finalImage[y:y+h, x:x+w]
            # cv2.imshow("selected rectangle:"+str(imageIndex), roiImage)


            # Histograms of the selected image for further processing 
            cv2.imwrite("plate"+str(imageIndex)+".jpg",roiImage)


# cv2.imshow("Contour image", contourImage)
cv2.imshow("Rectangle detected", rectImage)


#------------------------------------Character segmentation PROCESS-------------------------->
croppedImage = cv2.imread("plate"+ str(imageIndex)+".jpg")
'''croppedCopyTmg = croppedImage.copy()
preProcessing = preProcessingImage(croppedCopyTmg)
cleanBorder =cleanBoder(preProcessing,10)
areaFilter = checkArea(cleanBorder,160)
diluteFilter = fillCharacters(areaFilter)
charSegment = charcterSegment(diluteFilter,croppedImage,"top")
cv2.imshow("final segmentation",charSegment)
cv2.imshow("area Filter",areaFilter)
cv2.imshow("cleanBorder",cleanBorder)
cv2.imshow("pre",preProcessing)

'''



#-----------------------segmenting image into two halves------------>
print("###########################################################################")
print("Segmenting image into two halves...")
cImageWidth = croppedImage.shape[1]
cImageHeight = croppedImage.shape[0]
print("Total image height:", cImageHeight)
print("Total image width:", cImageWidth)



#----------------------------Top image----------------------------->
startRow, startCol = int(0), int(0)
endRow, endCol = int(cImageHeight*0.5), int(cImageWidth)
topImage = croppedImage[startRow:endRow, startCol:endCol]
# cv2.imshow("TopPart cropped", topImage)
preProcessingTop = preProcessingImage(topImage)
cleanBorderTop = cleanBorder(preProcessingTop, 5)
inpImgCopyAreaTop = cleanBorderTop[:, 55:cleanBorderTop.shape[1]-50]
checkAreaTop = checkArea(inpImgCopyAreaTop,120)
tempTop = topImage[:, 55:topImage.shape[1]-50]
# cv2.imshow("PreprocessTopPart",preProcessingTop)
# cv2.imshow("borderRemoval topPart",cleanBorderTop)
# cv2.imshow("areaFilter TopPart",checkAreaTop)
dilateTop = fillCharacters(checkAreaTop)
charSeg = characterSegment(dilateTop, tempTop, "top")

cv2.imshow("final topPart Image", charSeg)


#------Bottom Image----------->
startHeight,startWidth = int(cImageHeight*0.5),int(0)
endHeight,endWidth = int(cImageHeight),int(cImageWidth)
bottomimage = croppedImage[startHeight:endHeight,startWidth:endWidth]
#cv2.imshow("BottomPart cropped ",bottomimage)
preProcessingBottom = preProcessingImage(bottomimage)
cleanBorderBottom = cleanBorder(preProcessingBottom,0)
inpImgCopyAreaBot = cleanBorderBottom[:cleanBorderBottom.shape[0]-15,25:cleanBorderBottom.shape[1]-20]
checkAreaBottom = checkArea(inpImgCopyAreaBot,160)
tempBottom = bottomimage[:,25:bottomimage.shape[1]-25]
dilateBottom = fillCharacters(checkAreaBottom)
charsegBot = characterSegment(dilateBottom,tempBottom,"bottom")
#cv2.imshow("preprocess bottomPart",preProcessingBottom)
#cv2.imshow("borderCleaning bottomPart",cleanBorderBottom)
#cv2.imshow("areaFilter bottomPart",checkAreaBottom)
cv2.imshow("finalBottomPart",charsegBot)


#--------------------------------------------------->

c = cv2.waitKey(0)
if c== 27 :
    cv2.destroyAllWindows()