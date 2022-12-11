import cv2
import numpy as np

colour_list = ['b', 'r', 'g', 'y']
number_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'R', 'S', 'T']
comb_list = [(x + str(y)) for y in number_list for x in colour_list] #+ ['kB', 'kE', 'kW', 'kF']
print(comb_list)

color_list = [
    ['red', [0, 160, 70], [12, 250, 250]],
    ['yellow', [15, 50, 70], [33, 250, 250]],
    ['green', [34, 50, 70], [70, 250, 250]],
    ['blue', [100, 50, 70], [130, 250, 250]],
]

def detect_main_color(hsv_image, colors):
    color_found = 'undefined'
    max_count = 0

    for color_name, lower_val, upper_val in colors:
        # threshold the HSV image - any matching color will show up as white
        mask = cv2.inRange(hsv_image, np.array(lower_val), np.array(upper_val))

        # count white pixels on mask
        count = np.sum(mask)
        if count > max_count:
            color_found = color_name
            max_count = count

    return color_found

dataB = []
print('COLOURS BY PICTURE\n')
for x in range(len(comb_list)):
    img = cv2.imread(r'C:\Users\19pil\Documents\3rd Year Uni\Eris\images\c' + comb_list[x] + ".jpg")  # open the saved image in colour         
    img_crop =  img[290:390,260:390,:]
    
#----------------Colour detection code for cards in pictures--------------------------------
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    print(comb_list[x], detect_main_color(hsv, color_list))
#-------------------------------------------------------------------------------------------

    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)   # convert to B/W
    img_blur = cv2.GaussianBlur(img, (7,7),1)
    thr_value, img_th = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    img_close = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)      # morphology correction
    img_canny = cv2.Canny(img_close, 50, 100)                          # edge detection
    contours, hierarchy = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # extract contours on binarised image, not on canny
    cv2.drawContours(img_crop, contours, -1, (255,0,255), 1)        # paint contours on top of original coloured mage
    
    data = []
    for i, c in enumerate(contours):            # loop through all the found contours
        length = len(c)
        perimeter = cv2.arcLength(c, True)      # perimeter of contour c (curved length)
        epsilon = 0.02*perimeter    # parameter of polygon approximation: smaller values provide more vertices
        vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))     # approximate with polygon
        [x,y,w,h] = cv2.boundingRect(c)
        cv2.rectangle(img_crop, (x,y), (x+w,y+h), (255, 0, 0), 2)
        sample = [length, perimeter, vertex_approx]
        data.append(sample)
        
    BigCon = (max(data, key=lambda x: x[1]))    # with the same code we print the new biggest countour found, that is the number, letters or shapes that helps to identify the specific card
    dataB.append(BigCon)
    target = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 'R', 'R', 'R', 'R', 'S', 'S', 'S', 'S', 'T', 'T', 'T', 'T']
print('\n CONTOURS\n')
print(np.array(dataB))
print(np.array(target))

import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
d = dataB
t = target
clf = MLPClassifier(alpha=1, max_iter=2000)
d_train, d_test, t_train, t_test = train_test_split(d, t, test_size=.5, random_state=40)
clf.fit(d_train, t_train)
score = clf.score(d_test, t_test) # predict and calculate the classification accuracy
print(score*100, '%')

for i in range(len(t_test)):
    print(t_test[i], t_test[i])
    
filename = 'Uno_Classifier.sav'
joblib.dump(clf, filename)
