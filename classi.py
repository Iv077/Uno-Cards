import cv2
import numpy as np
import joblib

vc = cv2.VideoCapture(0)

classifier = joblib.load('Uno_Classifier.sav')

while True:
    classifier = joblib.load('Uno_Classifier.sav')
    success, img = vc.read()
    imgContour = img.copy()
    img_blur = cv2.GaussianBlur(img, (7,7),1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)   # convert to B/W
    
    img_canny = cv2.Canny(img_gray, 70, 190)    # standard canny edge detector
    kernel = np.ones((3, 3))
    img_dil = cv2.dilate(img_canny, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(img_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #contours is not an image, is a chain of pixel locations    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500 and area < 7000:
            lenght1 = len(cnt)
            cv2.drawContours(imgContour, cnt,-1,(255, 0, 255),2)
            perimeter1 = cv2.arcLength(cnt, True)      # perimeter of contour c (curved length)
            epsilon1 = 0.02*perimeter1    # parameter of polygon approximation: smaller values provide more vertices
            approx1 = len(cv2.approxPolyDP(cnt, epsilon1, True))
            [x,y,w,h] = cv2.boundingRect(cnt)
            cv2.rectangle(imgContour, (x,y), (x+w,y+h), (255, 0, 0), 2)
            parameter = [lenght1, perimeter1, approx1]
            d_test = [parameter]
            hello = classifier.predict(d_test)
            print(hello)
#             cv2.putText(imgContour, 'Points:' + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255,0),2)
#             cv2.putText(imgContour, 'Areas:' + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255,0),2)
       
    cv2.imshow('Result2', imgContour)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
vc.release()        
