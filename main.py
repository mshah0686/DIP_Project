import cv2
import numpy as np
import pre_process
import imutils
import tensorflow as tf

'''
Main file:
-> Read an image
-> Do preprocessing
-> Pass into digit classification network
-> Pass into soduku solver network
-> Return solution
'''

def localizeEquation(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expected_digits = []
    expected_digits_x_loc = []

    overlay_image = img.copy()
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    for c in contours:
        area = cv2.contourArea(c)
        if(area > 15):
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(overlay_image, (x,y), (x+w, y+h), (255, 0, 0), 2)
            sub_image = img[y:y+h, x:x+w]
            max_dimen = max(sub_image.shape[0], sub_image.shape[1]) + 10
            x_pad = ((max_dimen - sub_image.shape[0]) // 2)
            y_pad = ((max_dimen - sub_image.shape[1]) // 2) 
            sub_image = np.pad(sub_image, ((x_pad,x_pad), (y_pad,y_pad)), "constant", constant_values=0)
            sub_image = cv2.resize(sub_image, (28, 28))
            cv2.imshow("Area of interest", sub_image)
            cv2.waitKey(0)
            expected_digits.append(sub_image)
            expected_digits_x_loc.append(x)
    cv2.imshow("Rectangle", overlay_image)
    cv2.waitKey(0)

    #order the digits based on x axis: know ordering for equation
    X = np.array(expected_digits)
    Y = np.array(expected_digits_x_loc)
    inds = Y.argsort()
    sorted_list = X[inds]
    return sorted_list

if __name__ == "__main__":
    print("Reading Image")
    img = cv2.imread('sample3.jpg',0)
    img = imutils.resize(img, height = 400, width=400)
    print("Doing pre processing")
    img = pre_process.process_pipeline(img)
    areas_of_interest = np.asarray(localizeEquation(img))

    print("Found Areas of Interest: ", len(areas_of_interest))

    #Machine Learning Part
    print("Starting Machine Learning Part")
    #Reshape data
    x_test = areas_of_interest.reshape(areas_of_interest.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255

    # load model
    digit_model = tf.keras.models.load_model('model_data/model_ver_0')
    digit_model.summary()

    #predict on those two areas
    print("Running prediction")
    predictions = digit_model.predict(x_test)
    print("Predictions: ", predictions)
    characters = []
    for p in predictions:
        max_index = np.argmax(p)
        max_val = np.max(p)
        if(max_val > 0.50):
            characters.append(max_index)
    print(characters)