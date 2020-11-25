import cv2
import numpy as np
import imutils
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
    #Final process pipeline, used in main.py
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    img = cv2.bitwise_not(img)
    return img

def testing_pipeline(img):
    '''
    Use this for testing different filters
    '''
    #ret, img = cv2.threshold(img,90,255,cv2.THRESH_BINARY)
    #img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imshow("threshold", img)
    cv2.waitKey(0)
    kernel = np.ones((2,2),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)
    cv2.imshow("erode", img)
    cv2.waitKey(0)
    img = cv2.erode(img,kernel,iterations = 1)
    cv2.imshow("erode", img)
    cv2.waitKey(0)
    #img = cv2.dilate(img,kernel,iterations = 1)
    #ret, img = cv2.threshold(img,230,255,cv2.THRESH_BINARY)
    img = cv2.bitwise_not(img)
    cv2.imshow("erode", img)
    cv2.waitKey(0)
    #img = cv2.bilateralFilter(img,9,20,75)
    #cv2.imshow("erode", img)
    #cv2.waitKey(0)
    #img = cv2.medianBlur(img,3)
    #img = cv2.dilate(img,kernel,iterations = 1)
    return img

def smart_processing(img):
    '''
    Look at areas of interest, and remove all noise not in those areas
    I realize that this is really stupid actually, you're just finding contours twice..
    '''
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    img = cv2.bitwise_not(img)
    cv2.imshow("threshold", img)
    cv2.waitKey(0)
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_image = img.copy() * 0
    #overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    print("Total contours found: ", len(contours))
    for c in contours:
        area = cv2.contourArea(c)
        if(area > 10):
            (x,y,w,h) = cv2.boundingRect(c)
            overlay_image[y:y+h, x:x+w] = img[y:y+h, x:x+w]
            cv2.rectangle(overlay_image, (x,y), (x+w, y+h), (255, 0, 0), 2)
            sub_image = img[y:y+h, x:x+w]
            cv2.imshow("Area of interest", sub_image)
            cv2.waitKey(0)
    cv2.imshow("Rectangle", overlay_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    '''
    Will run testing filter pipeline
    '''
    print("Running main function")
    img = cv2.imread('sample3.jpg',0)
    img = imutils.resize(img, height = 400, width=400)
    #smart_processing(img)
    #processed = testing_pipeline(img)
    #cv2.imshow('original image',img)
    #cv2.imshow('processed image', processed)

    k = cv2.waitKey(0)
    while(k != 27):
        k = cv2.waitKey(0)
    
    cv2.destroyAllWindows()