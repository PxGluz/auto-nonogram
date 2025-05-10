import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pytesseract
from PIL import Image
import math

#### READ THE MODEL WEIGHTS
def intializePredectionModel():
    model = load_model('Resources/myModel.h5')
    return model


#### 1 - Preprocessing Image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold


#### 3 - Reorder points for Warp Perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


#### 3 - FINDING THE BIGGEST COUNTOUR ASSUING THAT IS THE SUDUKO PUZZLE
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

def findSizes(sourceImage, imWidth, imHeight):
    imageMatrix = np.array(sourceImage)
    startingI = imHeight - 2
    startingJ = imWidth - 2
    while not imageMatrix[startingI-1][startingJ] or not imageMatrix[startingI+1][startingJ] or not imageMatrix[startingI][startingJ-1] or not imageMatrix[startingI][startingJ+1]:
        if not imageMatrix[startingI-1][startingJ] or not imageMatrix[startingI+1][startingJ]:
            startingJ -= 1
        if not imageMatrix[startingI][startingJ-1] or not imageMatrix[startingI][startingJ+1]:
            startingI -= 1
    borderDimX = imWidth - startingJ
    borderDimY = imHeight - startingI
    i = startingI
    j = startingJ
    foundBlack = False
    while not imageMatrix[startingI-1][startingJ] or not imageMatrix[startingI+1][startingJ] or not imageMatrix[startingI][startingJ-1] or not imageMatrix[startingI][startingJ+1] or not foundBlack:
        print(foundBlack, imageMatrix[startingI][startingJ], imageMatrix[startingI-1][startingJ], imageMatrix[startingI+1][startingJ], imageMatrix[startingI][startingJ-1], imageMatrix[startingI][startingJ+1])
        if not foundBlack and imageMatrix[startingI][startingJ] == 0:
            foundBlack = True
        if not imageMatrix[startingI-1][startingJ] or not imageMatrix[startingI+1][startingJ] or not foundBlack:
            startingJ -= 1
        if not imageMatrix[startingI][startingJ-1] or not imageMatrix[startingI][startingJ+1] or not foundBlack:
            startingI -= 1


    return j - startingJ, i - startingI, borderDimX, borderDimY

#### 4 - TO SPLIT THE IMAGE INTO 81 DIFFRENT IMAGES
def splitBoxes(img, height, width, widthImg, heightImg):
    img = cv2.resize(img, (math.floor(widthImg / width) * width, math.floor(heightImg / height) * height))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
    rows = np.vsplit(img,height)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,width)
        for box in cols:
            boxes.append(box)
    return boxes


#### 4 - GET PREDECTIONS ON ALL IMAGES
def getPredection(boxes):
    result = []
    for image in boxes:
        ## PREPARE IMAGE
        # img = np.asarray(image)
        # img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        # img = cv2.resize(img, (28, 28))
        # img = img / 255
        # img = img.reshape(1, 28, 28, 1)
        ## GET PREDICTION
        im_pil = Image.fromarray(image)
        prediction = pytesseract.image_to_string(im_pil, lang='eng',config='--psm 9 --oem 3 -c tessedit_char_whitelist=0123456789')
        # probabilityValue = np.amax(predictions)
        ## SAVE TO RESULT
        
        if prediction == "":
            prediction = " "
        result.append(prediction.replace("\n", ""))
        # if probabilityValue > 0.8:
        #     result.append(classIndex[0])
        # else:
        #     result.append(-1)
    return result


#### 6 -  TO DISPLAY THE SOLUTION ON THE IMAGE
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


#### 6 - DRAW GRID TO SEE THE WARP PRESPECTIVE EFFICENCY (OPTIONAL)
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img


#### 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver