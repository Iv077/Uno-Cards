import cv2
import numpy as np
import joblib

vc = cv2.VideoCapture(0)

color_list = [
        ['red', [0, 160, 70], [12, 250, 250]],
        ['yellow', [15, 50, 70], [33, 250, 250]],
        ['green', [40, 50, 70], [70, 250, 250]],
        ['blue', [100, 50, 70], [130, 250, 250]]]
    

def detect_main_color(hsv, colors):
    color_found = 'undefined'
    max_count = 0

    for color_name, lower_val, upper_val in color_list:
        # threshold the HSV image - any matching color will show up as white
        mask = cv2.inRange(hsv, np.array(lower_val), np.array(upper_val))

        # count white pixels on mask
        count = np.sum(mask)
        if count > max_count:
            color_found = color_name
            max_count = count

    return color_found

while True:
    classifier = joblib.load('Uno_Classifier.sav')
    success, img = vc.read()
    imgContour = img.copy()
    img_blur = cv2.GaussianBlur(img, (7,7),1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)   # convert to B/W
    img_canny = cv2.Canny(img_gray, 50, 100)    # standard canny edge detector
    kernel = np.ones((5, 5), np.uint8)
    img_dil = cv2.dilate(img_canny, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(img_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #contours is not an image, is a chain of pixel locations    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500 and area < 7000:
            lenght1 = len(cnt)
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 1)
            perimeter1 = cv2.arcLength(cnt, True)      # perimeter of contour c (curved length)
            epsilon1 = 0.02*perimeter1    # parameter of polygon approximation: smaller values provide more vertices
            approx1 = len(cv2.approxPolyDP(cnt, epsilon1, True))
            [x,y,w,h] = cv2.boundingRect(cnt)
            cv2.rectangle(imgContour, (x,y), (x+w,y+h), (255, 0, 0), 2)
            parameter = [lenght1, perimeter1, approx1]
            d_test = [parameter]
            classi = classifier.predict(d_test)
            cv2.putText(imgContour, 'Card:' + str(classi), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255,0),2)
            dec = detect_main_color(img, color_list)
            cv2.putText(imgContour, 'Colour:' + str(dec), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255,0),2)
       
    cv2.imshow('Result', imgContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
vc.release()        
