import numpy as np
import cv2 as cv

# choose your camera
cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # blur = cv.GaussianBlur(gray, (5, 5),
    #                    cv.BORDER_DEFAULT)
    ret, thresh = cv.threshold(hsv, 200, 255,
                           cv.THRESH_BINARY_INV)

    # #green
    lower_limit = np.array([30, 100, 100])
    upper_limit = np.array([60, 255, 255])

    #blue  
    lower_limit = np.array([90, 50, 50])
    upper_limit = np.array([120, 255, 255])

    mask = cv.inRange(hsv, lower_limit, upper_limit)
    # 1 1 = 1
    # 0 1 = 0
    # 1 0 = 0
    # 0 0 = 1
    contours, hierarchies = cv.findContours(mask.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(mask.copy().shape[:2], dtype='uint8')


    result = cv.bitwise_and(frame, frame, mask=mask)
    cv.drawContours(blank, contours, -1, (255, 0, 0), 1)
    contour_list = []
    for i in contours:
        # determine wether there is ci
        approx = cv.approxPolyDP(i,0.01*cv.arcLength(i,True),True)
        area = cv.contourArea(i)
    if ((len(approx) > 8) & (area > 30) ):
        contour_list.append(i)
        M = cv.moments(i)
        # checking where is the center of the object
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv.drawContours(result, contour_list, 0, (0, 255, 0), 2)
            cv.circle(result, (cx, cy), 7, (0, 0, 255), -1)
            cv.putText(result, "center", (cx - 20, cy - 20),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# cx =   (M10 / M00 )
# cy =  ( M01 / M00 )


    cv.imshow('frame', result)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()