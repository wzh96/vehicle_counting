import cv2
import numpy as np
import pandas as pd
from centroidtracker import CentroidTracker
from check_intersect import Point
from check_intersect import doIntersect
import datetime

cap = cv2.VideoCapture('Standard_SCU3LM_2017-03-15_1600.002.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out_det.avi', fourcc, 60,(720,480))


tracker = CentroidTracker(maxDisappeared=2, maxDistance=90)


ret, frame1 = cap.read()
ret, frame2 = cap.read()

def draw_rect(rects,color):
    for r in rects:
        p1 = (r[0],r[1])
        p2 = (r[2],r[3])
        cv2.rectangle(frame1,p1,p2,color,4)
def draw_centroid(centroid,color):
    for r in centroid:
        center = (r[0],r[1])
        cv2.circle(frame1,center,3,color,-1)

object_list = []

#Draw reference line here
xo = 343
yo = 206
xd = 398
yd = 224

p2 = Point(xo,yo)
q2 = Point(xd,yd)

count = 0
time = str([])
output = []
while cap.isOpened():
    object_list = object_list[-20:]    
    rects = []
    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGRA2GRAY)
    blur = cv2.blur(gray,(5,5))
    _, thresh = cv2.threshold(blur,40,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (xo, yo), (xd, yd), (0, 0, 255),2)

    for i in range(len(contours)):
        if hierarchy[0, i, 3] == -1:
            if cv2.contourArea(contours[i]) < 1000:
                continue
            (x,y,w,h) = cv2.boundingRect(contours[i])

            rects.append([x, y, x+w, y+h])
    
    
    rects,weights = cv2.groupRectangles(rects, 0, 2.5)

    objects = tracker.update(rects)

    for (objectId, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        cv2.circle(frame1,(cx, cy),5,(0, 255, 0),-1)
        text = "ID: {}".format(objectId)
        cv2.putText(frame1, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        centroid = np.array([cx,cy])
        object_ID = np.array([objectId]) 
        
        object_list.append(np.concatenate((object_ID,centroid)))

    
    object_list_arr = np.asarray(object_list)
    uni_id = np.unique(object_list_arr[:,0])

    df = pd.DataFrame(object_list_arr,columns=['id','xc','yc'])
    for id in uni_id:
        coo = df.loc[df['id'] == id][['xc','yc']]
        current_coo = coo.tail(2)
        current_coo = current_coo.to_numpy()
        if np.shape(current_coo)[0] == 2:
            p1 = Point(current_coo[0,0],current_coo[0,1])
            q1 = Point(current_coo[1,0],current_coo[1,1])
            if doIntersect(p1, q1, p2, q2): 
                count = count + 1
                time = str(datetime.datetime.now())
                output.append([id,time])
            else:
                pass
        else:
            pass

    counter =  "Count: {}".format(count)
    cv2.putText(frame1, counter, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),2)
    timer = time
    cv2.putText(frame1, time,(10,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

     

    

    cv2.imshow('feed',frame1)
    out.write(frame1)

    frame1 = frame2

    ret, frame2 = cap.read()


    if cv2.waitKey(1) == 27:
        break

#output table 
output = np.asarray(output)
output = pd.DataFrame(output,columns=['vehicle_id','time pass reference line'])
output.to_csv('output_list.csv')

cv2.destroyAllWindows()
cap.release()



    # draw_rect(rects, (0,255,0))

    # if len(rects) != 0:
    #     cx = (rects[:,0]+rects[:,2])/2
    #     cy = (rects[:,1]+rects[:,3])/2
    #     centroid = np.column_stack((cx.astype(int),cy.astype(int)))
        
    #     draw_centroid(centroid,(0,0,255))
