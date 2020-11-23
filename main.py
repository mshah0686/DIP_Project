import cv2
import numpy as np
import pre_process
import imutils

'''
Main file:
-> Read an image
-> Do preprocessing
-> Pass into digit classification network
-> Pass into soduku solver network
-> Return solution
'''

def findSodukuLocation(img):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    puzzle_boundary_countour = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzleCnt = approx
            break
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay_image = img.copy()
    print(puzzleCnt)
    cv2.drawContours(overlay_image, [puzzleCnt], -1, (255, 0, 0), 3)
    cv2.imshow("Original", img)
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Puzzle Outline", overlay_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return [puzzleCnt]


if __name__ == "__main__":
    img = cv2.imread('sudoku_sample.jpg',0)
    img = pre_process.process_pipeline(img)
    findSodukuLocation(img)