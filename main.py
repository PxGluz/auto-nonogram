print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
# import sudukoSolver

########################################################################
pathImage = "Resources/1.jpg"
heightImg = 450
widthImg = 450
# model = intializePredectionModel()  # LOAD THE CNN MODEL
########################################################################


#### 1. PREPARE THE IMAGE
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgThreshold = preProcess(img)

# #### 2. FIND ALL COUNTOURS
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

#### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    # imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    thresholdPuzzle = preProcess(imgWarpColored)
    boxWidth, boxHeight, borderDimX, borderDimY = findSizes(thresholdPuzzle, widthImg, heightImg)
    matrixWidth = round((widthImg - 2 * borderDimX) / boxWidth)
    matrixHeight = round((heightImg - 2 * borderDimY) / boxHeight)

    print(matrixWidth, matrixHeight)

    imgSolvedDigits = imgBlank.copy()

    boxes = splitBoxes(imgWarpColored, matrixHeight, matrixWidth, widthImg, heightImg)
    print(len(boxes))
    # cv2.imshow("Sample",boxes[5])
    numbers = getPredection(boxes)
    numbers = np.reshape(numbers, (matrixHeight, matrixWidth))
    print(numbers)


imageArray = ([img,imgThreshold,imgContours, imgBigContour],
                [imgWarpColored, thresholdPuzzle,imgBlank,imgBlank])
stackedImage = stackImages(imageArray, 1)
cv2.imshow('Stacked Images', thresholdPuzzle)
cv2.waitKey(0)