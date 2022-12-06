import cv2
import numpy as np

colour_list = ['b', 'r', 'g', 'y']
number_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'R', 'S', 'T']
comb_list = [(x + str(y)) for y in number_list for x in colour_list] #+ ['ckB', 'ckE', 'ckW', 'ckF']
print(comb_list)

dataB = []
for x in range(len(comb_list)):
#    print(r'C:\Users\19pil\Documents\3rd Year Uni\Eris\images\c' + comb_list[x] + ".jpg")
    imgc = cv2.imread(r'C:\Users\19pil\Documents\3rd Year Uni\Eris\images\c' + comb_list[x] + ".jpg")  # open the saved image in colour         
    img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)   # convert to B/W
    img_sm = cv2.blur(img, (3, 3))           # smoothing
    thr_value, img_th = cv2.threshold(img_sm, 0, 255, cv2.THRESH_OTSU)   # binarisation
    kernel = np.ones((5, 5), np.uint8)
    img_close = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)      # morphology correction
    img_canny = cv2.Canny(img_close, 50, 100)                          # edge detection
    contours, hierarchy = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # extract contours on binarised image, not on canny
    cv2.drawContours(imgc, contours, -1, (0, 255, 0), 1)         # paint contours on top of original coloured mage
    data = []
    for i, c in enumerate(contours):            # loop through all the found contours
        length = len(c)
        perimeter = cv2.arcLength(c, True)      # perimeter of contour c (curved length)
        epsilon = 0.02*perimeter    # parameter of polygon approximation: smaller values provide more vertices
        vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))     # approximate with polygon
        cv2.drawContours(imgc, [c], 0, (0, 255, 0), 3)   # paint contour c
        cv2.putText(imgc, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))    # identify contour c
        [x,y,w,h] = cv2.boundingRect(c)
        cv2.rectangle(imgc, (x,y), (x+w,y+h), (255, 0, 0), 2)
        sample = [length, perimeter, vertex_approx]
        data.append(sample)
    data.remove(max(data, key=lambda x: x[1]))     # removing the biggest contour found, in our case the border of the uno card
    BigCon = (max(data, key=lambda x: x[1]))    # with the same code we print the new biggest countour found, that is the number, letters or shapes that helps to identify the specific card
    dataB.append(BigCon)
    tar = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 'R', 'R', 'R', 'R', 'S', 'S', 'S', 'S', 'T', 'T', 'T', 'T']
    t = np.array(tar)
print(np.array(dataB))
print(t)

d = dataB
#t = tar

# taking the first 60% of each class for training, and the rest 40% for testing
d_train = np.concatenate((d[:3,:], d[4:7,:], d[8:11,:], d[12:15,:], d[16:19,:], d[20:23,:], d[24:27,:], d[28:31,:], d[32:35,:], d[36:39,:], d[40:43,:], d[44:47,:], d[48:51,:]))
d_test = np.concatenate((d[3:4,:], d[7:8,:], d[11:12,:], d[15:16,:], d[19:20,:], d[23:24,:], d[27:28,:], d[31:32,:], d[35:36,:], d[39:40,:], d[43:44,:], d[47:48,:], d[51:,:]))
t_train = np.concatenate((t[:3], t[4:7], t[8:11], t[12:15], t[16:19], t[20:23], t[24:27], t[28:31], t[32:35], t[36:39], t[40:43], t[44:47], t[48:51]))
t_test = np.concatenate((t[3:4], t[7:8], t[11:12], t[15:16], t[19:20], t[23:24], t[27:28], t[31:32], t[35:36], t[39:40], t[43:44], t[47:48], t[51:]))

print(np.shape(d_train), np.shape(d_test), len(t_train), len(t_test))

print(d_train)
print(d_test)
print(t_train)
print(t_test)


