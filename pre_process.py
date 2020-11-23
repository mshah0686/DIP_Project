import cv2
import numpy as np
'''
Take a soduku image, and return a clean mask after processing
'''

'''
Notes:

Thresholding: ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
Histogram Equalization: img = cv2.equalizeHist(img)
Guassian Thresholding = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
Mean Thresholding: cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

Dilation: cv2.dilate(img,kernel,iterations = 1)         kernel = np.ones((5,5),np.uint8)
Erode:  cv2.erode(img,kernel,iterations = 1)

Median filter:  cv2.medianBlur(img,5)
Bilateral Filter: cv2.bilateralFilter(img,9,75,75)
Gaussian Blur: cv2.GaussianBlur(img,(5,5),0)

'''

def process_pipeline(img):
    #ret, img = cv2.threshold(img,90,255,cv2.THRESH_BINARY)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    kernel = np.ones((2,2),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)
    img = cv2.erode(img,kernel,iterations = 1)
    img = cv2.dilate(img,kernel,iterations = 1)
    img = cv2.GaussianBlur(img,(5,5),0)
    ret, img = cv2.threshold(img,230,255,cv2.THRESH_BINARY)
    return img

if __name__ == "__main__":
    print("Running main function")
    img = cv2.imread('sudoku_sample.jpg',0)
    processed = process_pipeline(img)
    cv2.imshow('original image',img)
    cv2.imshow('processed image', processed)

    k = cv2.waitKey(0)
    while(k != 27):
        k = cv2.waitKey(0)
    
    cv2.destroyAllWindows()