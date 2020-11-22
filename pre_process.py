import cv2 as cv
'''
Take a soduku image, and return a clean mask after processing
'''

def histogram_equalization(img):
    return cv.equalizeHist(img)

def dilate(img, kernel, iterations=1):
    return cv2.dilate(img,kernel,iterations = 1)

def erosion(img, kernel, iterations=1):
    return cv2.erode(img, kernel, iterations)

def threshold(low = 127, upper=255):
    return cv.threshold(img,low,upper,cv.THRESH_BINARY)

# TODO: use this for thresholding: OTSU's binarazation: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

def create_mask(img):
    #TODO: Main method for filtering and masking

