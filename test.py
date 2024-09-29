# imports:
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading an image in default mode:
inputImage = cv2.imread('shuidao/20240914141618.jpg')
# Deep copy for results:
inputImageCopy = inputImage.copy()

# Get local maximum:
kernelSize = 30
maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
localMax = cv2.morphologyEx(inputImage, cv2.MORPH_CLOSE, maxKernel, None, None, 1, cv2.BORDER_REFLECT101)

# Perform gain division
gainDivision = np.where(localMax == 0, 0, (inputImage/localMax))

# Clip the values to [0,255]
gainDivision = np.clip((255 * gainDivision), 0, 255)

# Convert the mat type from float to uint8:
gainDivision = gainDivision.astype("uint8")

# Convert RGB to grayscale:
grayscaleImage = cv2.cvtColor(gainDivision, cv2.COLOR_BGR2GRAY)

# Get binary image via Otsu:
_, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find the blobs on the binary image:
contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Store the bounding rectangles here:
rectanglesList = []

# Look for the outer bounding boxes (no children):
for _, c in enumerate(contours):

    # Get blob area:
    currentArea = cv2.contourArea(c)
    # Set a min area threshold:
    minArea = 100

    if currentArea > minArea:

        # Approximate the contour to a polygon:
        contoursPoly = cv2.approxPolyDP(c, 3, True)
        # Get the polygon's bounding rectangle:
        boundRect = cv2.boundingRect(contoursPoly)

        # Store rectangles in list:
        rectanglesList.append(boundRect)

        # Get the dimensions of the bounding rect:
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]

        # Set bounding rect:
        # color = (0, 0, 255)
        # cv2.rectangle( inputImageCopy, (int(rectX), int(rectY)),
        #            (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2 )

        # cv2.imshow("Rectangles", inputImageCopy)
        # cv2.waitKey(0)

        print("Elements found: "+str(len(rectanglesList)))

