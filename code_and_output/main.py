# import the necessary packages
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import pandas as pd

files = glob.glob('output/*.png')
for f in files:
    os.remove(f)

from sort import *
cross_check = []
tracker = Sort()
memory = {}
time_test = {}
time_for_speed = []
df = pd.DataFrame(columns= ["TrackingID","FrameID","LaneID"])
df4 = pd.DataFrame(columns= ["TrackingID","Speed"])
#df = df.T
dict_id_speed = {}
line1 = [(2370,840), (400, 800)]
line2 = [(2550,820), (670, 780)]
line3 = [(2650,1120), (400, 1090)]
line4 = [(2650,1530), (500, 1520)]
line_speed_start = [(2650,1320), (100, 1260)]
line_speed_end = [(2650,970), (250, 940)]

counter1 = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video")
ap.add_argument("-o", "--output", required=True,
    help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.40,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.40,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


f1, g1 = 1045, 199 
f2, g2 = 2400, 199 
f3, g3 = 0, 1150 
f4, g4 = 2400, 2160

top_left_x = min([f1,f2,f3,f4])
top_left_y = min([g1,g2,g3,g4])
bot_right_x = max([f1,f2,f3,f4])
bot_right_y = max([g1,g2,g3,g4])
##################################################################################################
## You can crop the region of interest to ignore unnecessary detections

#def crop(frame):
#    x1, y1 = 1045, 199 
#    x2, y2 = 2100, 199 
#    x3, y3 = 0, 1150 
#    x4, y4 = 2300, 2160
#
#    top_left_x = min([x1,x2,x3,x4])
#    top_left_y = min([y1,y2,y3,y4])
#    bot_right_x = max([x1,x2,x3,x4])
#    bot_right_y = max([y1,y2,y3,y4])
#    
#    croped = frame[top_left_y:bot_right_y, top_left_x:bot_right_x]
#
##    cv2.imwrite("/home/bamwani/Desktop/ALL_ASSIGNMENTS/LOGIKLE/Source Code/output/frame-30222.png",co)
#    
#    return(croped)
#    
#def overlay(croped):
#    x1, y1 = 1045, 199 
#    x2, y2 = 2100, 199 
#    x3, y3 = 0, 1150 
#    x4, y4 = 2300, 2160
#
#    top_left_x = min([x1,x2,x3,x4])
#    top_left_y = min([y1,y2,y3,y4])
#    
#    x_offset=top_left_x
#    y_offset=top_left_y
#    frame[y_offset:y_offset+croped.shape[0], x_offset:x_offset+croped.shape[1]] = croped
#    return(frame)
#####################################################################################################

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"] , "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
#np.random.seed(42)
#COLORS = "0,255,0"
#COLORS = np.random.randint(0, 255, size=(200, 3),
#    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])

writer = None
(W, H) = (None, None)

frameIndex = 0


# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    cv2.putText(frame, "LANES ARE NUMBERED FROM LEFT TO RIGHT STARTING FROM 1", (100,50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 5)
#    if frameIndex<=1173 or 1185<= frameIndex<1870 or 2025>=frameIndex>1876 or 2859>=frameIndex>=2026:
    if frameIndex<=300 or 2025>frameIndex>1876: #300
        frameIndex += 1
        continue
    elif 300 < frameIndex < 1184: 
#    elif 1173 < frameIndex < 1184: 
        # if the frame was not grabbed, then we have reached the end
        # of the stream
#        line_speed_start = [(2650,1320), (400, 1240)]
#        line_speed_end = [(2650,970), (450, 940)]
        if not grabbed:
            break
    #1876 - 2025 - 2861
        # if the frame dimensions are empty, grab them
    #    croped = crop(frame)

        lane1 = np.array([[[180, 2159], [1570, 200], [1420, 200], [0, 1680]]], np.int32)
        lane2 = np.array([[[180, 2159], [1045, 2159], [1730, 200], [1570, 200]]], np.int32)
        lane3 = np.array([[[1730,200], [1860,200], [1855, 2159], [1045, 2159]]], np.int32)
        lane4 = np.array([[[2060,200], [1860, 200], [1860, 2159], [2880, 2159]]], np.int32)
        cv2.polylines(frame, [lane1], True, (0,255,0), thickness=1)
        cv2.polylines(frame, [lane2], True, (0,0,255), thickness=1)
        cv2.polylines(frame, [lane3], True, (0,255,0), thickness=1)
        cv2.polylines(frame, [lane4], True, (0,0,255), thickness=1)
        cv2.line(frame, line_speed_start[0], line_speed_start[1], (255, 255, 255), 1)
        cv2.line(frame, line_speed_end[0], line_speed_end[1], (255, 255, 255), 1)
        
    
        
        if W is None or H is None:
            (H, W) = frame.shape[:2]
    
    #    frame = adjust_gamma(frame, gamma=1.5)
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
    
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        center = []
        confidences = []
        classIDs = []
    
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
    
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
    
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    center.append(int(centerY))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    
    
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
        #print("idxs", idxs)
        #print("boxes", boxes[i][0])
        #print("boxes", boxes[i][1])
        
        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x+w, y+h, confidences[i]])
                #print(confidences[i])
                #print(center[i])
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)
        
        boxes = []
        indexIDs = []
        c = []
        
        previous = memory.copy()
        #print("centerx",centerX)
        #  print("centery",centerY)
        memory = {}
    
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]
    
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                
                if (int(top_left_x) <= int(box[0]) <= int(bot_right_x)):
                    if 155< (int(box[2])-int(box[0])) < 600 and (int(box[3])-int(box[1]))<650:
#                        if 155< (int(box[2])-int(box[0])) < 600 and (int(box[3])-int(box[1]))<650:
                        # extract the bounding box coordinates
                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))
            
                        # draw a bounding box rectangle and label on the image
                        # color = [int(c) for c in COLORS[classIDs[i]]]
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        ct1 = cv2.pointPolygonTest(lane1, p0, False)
                        ct2 = cv2.pointPolygonTest(lane2, p0, False)
                        ct3 = cv2.pointPolygonTest(lane3, p0, False)
                        ct4 = cv2.pointPolygonTest(lane4, p0, False)
                            
                        color = (255,0,0) if ct1==1 else (0,255,0) if ct2==1 else (255,0,255) if ct3==1 else (0,255,255) if ct4==1 else (0,0,255)
                        cv2.rectangle(frame, (x, y), (w, h), color, 4)
            
                        if indexIDs[i] in previous:
                            previous_box = previous[indexIDs[i]]
                            (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                            (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                            p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                            p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                            cv2.line(frame, p0, p1, color, 3)
#                            p2 = (int(10+x2 + (w2-x2)/2), int(10+y2 + (h2-y2)/2))
#                            cv2.putText(frame, str(p1), p2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 5)
#                            cv2.putText(frame, str(p0), p0, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 5)
                            id = indexIDs[i]    
                            #########################################################

                            ## CALIBRATION FOR FINDING MIN TIME TAKEN BY ANY CAR IN-
                            # -FIRST PART OF VIDEO TO CROSS THE TWO CROSS THE TWO LINES
                            ## ASSUMING THAT THE FASTEST CAR IS DRIVING AT THE SPEED LIMIT (CONSIDERED 40Km/h here)
                            if intersect(p0,p1,line_speed_start[0],line_speed_start[1]):
                                time_start = np.round(time.time(),3)
#                                print("ts",time_start)
                                time_test.update({id:time_start})
                            elif intersect(p0,p1,line_speed_end[0],line_speed_end[1]):
                                if id in time_test:
                                    time_taken = np.round(time.time(),3)-time_test.get(id)
                                    time_for_speed.append(time_taken)
                                    speed = 356/time_taken
                                    df3 = pd.DataFrame([[id,speed]], columns= ["TrackingID","Speed"])
#                                    df4=df4.append(df3,ignore_index=True)
                                    df4=df4.append(df3,ignore_index=True)
#                                    dict_id_speed['TrackingID'] = id
#                                    dict_id_speed['Speed'] = speed
                                    del time_test[id]
                                    
                            #########################################################
                                
                                
                            
        #                    print(indexIDs[i-1])
                            cv2.putText(frame, str(id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 5)
                            if intersect(p0, p1, line1[0], line1[1]) and indexIDs[i] not in cross_check:
                                counter1 += 1
                                cross_check.append(indexIDs[i])
                                lane = 2 if ct1==1 else 3 if ct2==1 else 4 if ct3==1 else 5 if ct4==1 else 1
                                df2 = pd.DataFrame([[id,frameIndex,lane]], columns= ["TrackingID","FrameID","LaneID"])
                                df = df.append(df2,ignore_index=True)
#                                df
#                                df2
            #                if intersect(p0, p1, line2[0], line2[1]):
            #                    counter2 += 1
            
                        # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        #text = "{}".format(indexIDs[i])
                        #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #                frame = overlay(croped)
                i += 1
    
    
        # draw line
    #    cv2.line(frame, line1[0], line1[1], (255, 0, 255), 2)
    #    cv2.line(frame, line2[0], line2[1], (255, 0, 255), 2)
    #    cv2.line(frame, line3[0], line3[1], (255, 0, 255), 2)
        cv2.line(frame, line1[0], line1[1], (0, 255, 255), 3)
        ##############################################
        #FOR CALIBRATION
        ##############################################
#        try:
#            min_time = min(time_for_speed)
#        except:
#            min_time = ("wait for it")
#        note_text = "current min time taken{}".format(min_time)
#        cv2.putText(frame, note_text, (50,110), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 6)
        # draw counter
        counter_text = "counter:{}".format(counter1)
        cv2.putText(frame, counter_text, (100,250), cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 0, 255), 7)
    #    cv2.putText(frame, "ctr2",str(counter2), (100,400), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 0, 255), 10)
        # counter += 1
###############################################################################################        
###############################################################################################
###############################################################################################
#    elif 1870<=frameIndex<= 1876:
    elif 1184<=frameIndex<= 1876:
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
    #1876 - 2025 - 2861
        # if the frame dimensions are empty, grab them
    #    croped = crop(frame)
        
        lane1 = np.array([[[330, 2159], [1790, 200], [1660, 200], [0, 1750]]], np.int32)
        lane2 = np.array([[[330, 2159], [1220, 2159], [1950, 200], [1790, 200]]], np.int32)
        lane3 = np.array([[[1950,200], [2090,200], [2020, 2159], [1220, 2159]]], np.int32)
        lane4 = np.array([[[2280,200], [2090, 200], [2020, 2159], [2980, 2159]]], np.int32)
        cv2.polylines(frame, [lane1], True, (0,255,0), thickness=1)
        cv2.polylines(frame, [lane2], True, (0,0,255), thickness=1)
        cv2.polylines(frame, [lane3], True, (0,255,0), thickness=1)
        cv2.polylines(frame, [lane4], True, (0,0,255), thickness=1)
        cv2.line(frame, line_speed_start[0], line_speed_start[1], (255, 255, 255), 1)
        cv2.line(frame, line_speed_end[0], line_speed_end[1], (255, 255, 255), 1)
    
        
        if W is None or H is None:
            (H, W) = frame.shape[:2]
    
    #    frame = adjust_gamma(frame, gamma=1.5)
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
    
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        center = []
        confidences = []
        classIDs = []
    
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
    
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
    
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    center.append(int(centerY))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    
    
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
        #print("idxs", idxs)
        #print("boxes", boxes[i][0])
        #print("boxes", boxes[i][1])
        
        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x+w, y+h, confidences[i]])
                #print(confidences[i])
                #print(center[i])
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)
        
        boxes = []
        indexIDs = []
        c = []
        
        previous = memory.copy()
        #print("centerx",centerX)
        #  print("centery",centerY)
        memory = {}
    
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]
    
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                
                if (int(top_left_x) <= int(box[0]) <= int(bot_right_x)):
                    if 155< (int(box[2])-int(box[0])) < 600 and (int(box[3])-int(box[1]))<650:
                        # extract the bounding box coordinates
                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))
            
                        # draw a bounding box rectangle and label on the image
                        # color = [int(c) for c in COLORS[classIDs[i]]]
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        ct1 = cv2.pointPolygonTest(lane1, p0, False)
                        ct2 = cv2.pointPolygonTest(lane2, p0, False)
                        ct3 = cv2.pointPolygonTest(lane3, p0, False)
                        ct4 = cv2.pointPolygonTest(lane4, p0, False)
                            
                        color = (255,0,0) if ct1==1 else (0,255,0) if ct2==1 else (255,0,255) if ct3==1 else (0,255,255) if ct4==1 else (0,0,255)
                        cv2.rectangle(frame, (x, y), (w, h), color, 4)
            
                        if indexIDs[i] in previous:
                            previous_box = previous[indexIDs[i]]
                            (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                            (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                            p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                            p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                            cv2.line(frame, p0, p1, color, 3)
                            id = indexIDs[i]
                            if intersect(p0,p1,line_speed_start[0],line_speed_start[1]):
                                time_start = np.round(time.time(),3)
#                                print("ts",time_start)
                                time_test.update({id:time_start})
                            elif intersect(p0,p1,line_speed_end[0],line_speed_end[1]):
                                if id in time_test:
                                    time_taken = np.round(time.time(),3)-time_test.get(id)
                                    time_for_speed.append(time_taken)
                                    speed = 460/time_taken # S=D/T speed limit = distance between 2 lnes/ time taken by car which is driving at speed limit
                                    df3 = pd.DataFrame([[id,speed]], columns= ["TrackingID","Speed"])
                                    df4=df4.append(df3,ignore_index=True)
#                                    dict_id_speed.update({"TrackingID":id})
#                                    dict_id_speed.update({"Speed":speed})
                                    del time_test[id]
            
                            if intersect(p0, p1, line2[0], line2[1]) and indexIDs[i] not in cross_check:
                                counter1 += 1
                                cross_check.append(indexIDs[i])
                                lane = 2 if ct1==1 else 3 if ct2==1 else 4 if ct3==1 else 5 if ct4==1 else 1
                                df2 = pd.DataFrame([[id,frameIndex,lane]], columns= ["TrackingID","FrameID","LaneID"])
                                df = df.append(df2,ignore_index=True)
                                
            #                if intersect(p0, p1, line2[0], line2[1]):
            #                    counter2 += 1
            
                        # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        #text = "{}".format(indexIDs[i])
                        #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #                frame = overlay(croped)
                i += 1
    
    
        # draw line
    #    cv2.line(frame, line1[0], line1[1], (255, 0, 255), 2)
    #    cv2.line(frame, line2[0], line2[1], (255, 0, 255), 2)
    #    cv2.line(frame, line3[0], line3[1], (255, 0, 255), 2)
        cv2.line(frame, line2[0], line2[1], (0, 255, 255), 3)
        ##############################################
        #FOR CALIBRATION
        ##############################################
#        try:
#            min_time = min(time_for_speed)
#        except:
#            min_time = ("wait for it")
#        note_text = "current min time taken{}".format(min_time)
#        cv2.putText(frame, note_text, (50,110), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 6)
    
#        note_text = "NOTE: Vehicle speeds are calibrated only at yellow line. speed of cars are more stable."
#        cv2.putText(frame, note_text, (50,110), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
        # draw counter
        counter_text = "counter:{}".format(counter1)
        cv2.putText(frame, counter_text, (100,250), cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 0, 255), 7)
    #    cv2.putText(frame, "ctr2",str(counter2), (100,400), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 0, 255), 10)
        # counter += 1
############################################################################################
############################################################################################
############################################################################################        
    elif 2025 < frameIndex < 2861:
#    elif 2859 < frameIndex < 2861:
                # if the frame was not grabbed, then we have reached the end
        # of the stream
        line_speed_start = [(2650,1530), (350, 1520)]
        line_speed_end = [(2650,1330), (550, 1320)]
        if not grabbed:
            break
    #1876 - 2025 - 2861
        # if the frame dimensions are empty, grab them
    #    croped = crop(frame)

        lane1 = np.array([[[1730, 600], [1910, 600], [780, 2159], [30, 2159]]], np.int32)
        lane2 = np.array([[[1910, 600], [2150, 600], [1600, 2159], [780, 2159]]], np.int32)
        lane3 = np.array([[[2150,600], [2350,600], [2340, 2159], [1600, 2159]]], np.int32)
        lane4 = np.array([[[2350,600], [2460, 600], [3140, 2159], [2340, 2159]]], np.int32)
#        lane5 = np.array([[[2260,600], [2260, 600], [2240, 2159], [2200, 2159]]], np.int32)
        cv2.polylines(frame, [lane1], True, (0,255,0), thickness=1)
        cv2.polylines(frame, [lane2], True, (0,0,255), thickness=1)
        cv2.polylines(frame, [lane3], True, (0,255,0), thickness=1)
        cv2.polylines(frame, [lane4], True, (0,0,255), thickness=1)
        cv2.line(frame, line_speed_start[0], line_speed_start[1], (255, 255, 255), 1)
        cv2.line(frame, line_speed_end[0], line_speed_end[1], (255, 255, 255), 1)
    
        
        if W is None or H is None:
            (H, W) = frame.shape[:2]
    
    #    frame = adjust_gamma(frame, gamma=1.5)
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
    
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        center = []
        confidences = []
        classIDs = []
    
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
    
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
    
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    center.append(int(centerY))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    
    
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
        #print("idxs", idxs)
        #print("boxes", boxes[i][0])
        #print("boxes", boxes[i][1])
        
        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x+w, y+h, confidences[i]])
                #print(confidences[i])
                #print(center[i])
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)
        
        boxes = []
        indexIDs = []
        c = []
        
        previous = memory.copy()
        #print("centerx",centerX)
        #  print("centery",centerY)
        memory = {}
    
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]
    
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                
                if (int(top_left_x) <= int(box[0]) <= int(bot_right_x)):
                    if 155< (int(box[2])-int(box[0])) < 600 and (int(box[3])-int(box[1]))<650:
                        # extract the bounding box coordinates
                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))
            
                        # draw a bounding box rectangle and label on the image
                        # color = [int(c) for c in COLORS[classIDs[i]]]
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        ct1 = cv2.pointPolygonTest(lane1, p0, False)
                        ct2 = cv2.pointPolygonTest(lane2, p0, False)
                        ct3 = cv2.pointPolygonTest(lane3, p0, False)
                        ct4 = cv2.pointPolygonTest(lane4, p0, False)
                            
                        color = (255,0,0) if ct1==1 else (0,255,0) if ct2==1 else (255,0,255) if ct3==1 else (0,255,255) if ct4==1 else (0,0,255)
                        cv2.rectangle(frame, (x, y), (w, h), color, 4)
            
                        if indexIDs[i] in previous:
                            previous_box = previous[indexIDs[i]]
                            (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                            (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                            p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                            p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                            cv2.line(frame, p0, p1, color, 3)
                            id = indexIDs[i]
                            
                            if intersect(p0,p1,line_speed_start[0],line_speed_start[1]):
                                time_start = np.round(time.time(),3)
#                                print("ts",time_start)
                                time_test.update({id:time_start})
                            elif intersect(p0,p1,line_speed_end[0],line_speed_end[1]):
                                if id in time_test:
                                    time_taken = np.round(time.time(),3)-time_test.get(id)
                                    time_for_speed.append(time_taken)
                                    speed = 920/time_taken
                                    df3 = pd.DataFrame([[id,speed]], columns= ["TrackingID","Speed"])
                                    df4=df4.append(df3,ignore_index=True)
#                                    dict_id_speed.update({"TrackingID":id})
#                                    dict_id_speed.update({"Speed":speed})
                                    del time_test[id]
                                    
                            cv2.putText(frame, str(id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 5)
        
                            
                            
            
                            if intersect(p0, p1, line3[0], line3[1]) and indexIDs[i] not in cross_check:
                                counter1 += 1
                                cross_check.append(indexIDs[i])
                                lane = 2 if ct1==1 else 3 if ct2==1 else 4 if ct3==1 else 5 if ct4==1 else 1
                                df2 = pd.DataFrame([[id,frameIndex,lane]], columns= ["TrackingID","FrameID","LaneID"])
                                df = df.append(df2,ignore_index=True)
                                
            #                if intersect(p0, p1, line2[0], line2[1]):
            #                    counter2 += 1
            
                        # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        #text = "{}".format(indexIDs[i])
                        #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #                frame = overlay(croped)
                i += 1
    
    
        # draw line
    #    cv2.line(frame, line1[0], line1[1], (255, 0, 255), 2)
    #    cv2.line(frame, line2[0], line2[1], (255, 0, 255), 2)
    #    cv2.line(frame, line3[0], line3[1], (255, 0, 255), 2)
        cv2.line(frame, line3[0], line3[1], (0, 255, 255), 3)
        ##############################################
        #FOR CALIBRATION
        ##############################################
#        try:
#            min_time = min(time_for_speed)
#        except:
##            min_time = ("wait for it")
#        note_text = "current min time taken{}".format(min_time)
#        cv2.putText(frame, note_text, (50,110), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 6)
    
#        note_text = "NOTE: Vehicle speeds are calibrated only at yellow line. speed of cars are more stable."
#        cv2.putText(frame, note_text, (50,110), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
        # draw counter
        counter_text = "counter:{}".format(counter1)
        cv2.putText(frame, counter_text, (100,250), cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 0, 255), 7)
    #    cv2.putText(frame, "ctr2",str(counter2), (100,400), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 0, 255), 10)
        # counter += 1
#####################################################################################################
#####################################################################################################
#####################################################################################################
    else:
        
        line_speed_start = [(2650,1530), (350, 1520)]
        line_speed_end = [(2650,1330), (550, 1320)]
        
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        
        if not grabbed:
            break
            frame = adjust_gamma(frame)
    #1876 - 2025 - 2861
        # if the frame dimensions are empty, grab them
    #    croped = crop(frame)
        lane1 = np.array([[[1495, 850], [1690, 850], [780, 2159], [80, 2159]]], np.int32)
        lane2 = np.array([[[1690, 850], [1920, 850], [1450, 2159], [780, 2159]]], np.int32)
        lane3 = np.array([[[1920,850], [2130,850], [2150, 2159], [1450, 2159]]], np.int32)
        lane4 = np.array([[[2130,850], [2300, 850], [2870, 2159], [2150, 2159]]], np.int32)
        cv2.polylines(frame, [lane1], True, (0,255,0), thickness=1)
        cv2.polylines(frame, [lane2], True, (0,0,255), thickness=1)
        cv2.polylines(frame, [lane3], True, (0,255,0), thickness=1)
        cv2.polylines(frame, [lane4], True, (0,0,255), thickness=1)
        cv2.line(frame, line_speed_start[0], line_speed_start[1], (255, 255, 255), 1)
        cv2.line(frame, line_speed_end[0], line_speed_end[1], (255, 255, 255), 1)
    
        
        if W is None or H is None:
            (H, W) = frame.shape[:2]
    
        frame = adjust_gamma(frame, gamma=1.5)
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
    
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        center = []
        confidences = []
        classIDs = []
    
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
    
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
    
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    center.append(int(centerY))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    
    
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
        #print("idxs", idxs)
        #print("boxes", boxes[i][0])
        #print("boxes", boxes[i][1])
        
        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x+w, y+h, confidences[i]])
                #print(confidences[i])
                #print(center[i])
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)
        
        boxes = []
        indexIDs = []
        c = []
        
        previous = memory.copy()
        #print("centerx",centerX)
        #  print("centery",centerY)
        memory = {}
    
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]
    
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                
                if (int(top_left_x) <= int(box[0]) <= int(bot_right_x)):
                    if 155< (int(box[2])-int(box[0])) < 600 and (int(box[3])-int(box[1]))<650:
                        # extract the bounding box coordinates
                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))
            
                        # draw a bounding box rectangle and label on the image
                        # color = [int(c) for c in COLORS[classIDs[i]]]
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        ct1 = cv2.pointPolygonTest(lane1, p0, False)
                        ct2 = cv2.pointPolygonTest(lane2, p0, False)
                        ct3 = cv2.pointPolygonTest(lane3, p0, False)
                        ct4 = cv2.pointPolygonTest(lane4, p0, False)
                            
                        color = (255,0,0) if ct1==1 else (0,255,0) if ct2==1 else (255,0,255) if ct3==1 else (0,255,255) if ct4==1 else (0,0,255)
                        cv2.rectangle(frame, (x, y), (w, h), color, 4)
            
                        if indexIDs[i] in previous:
                            previous_box = previous[indexIDs[i]]
                            (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                            (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                            p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                            p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                            cv2.line(frame, p0, p1, color, 3)
                            id = indexIDs[i]
                            
                            if intersect(p0,p1,line_speed_start[0],line_speed_start[1]):
                                time_start = np.round(time.time(),3)
#                                print("ts",time_start)
                                time_test.update({id:time_start})
                            elif intersect(p0,p1,line_speed_end[0],line_speed_end[1]):
                                if id in time_test:
                                    time_taken = np.round(time.time(),3)-time_test.get(id)
                                    time_for_speed.append(time_taken)
                                    speed = 701/time_taken
                                    df3 = pd.DataFrame([[id,speed]], columns= ["TrackingID","Speed"])
                                    df4=df4.append(df3,ignore_index=True)
#                                    dict_id_speed.update({"TrackingID":id})
#                                    dict_id_speed.update({"Speed":speed})
                                    del time_test[id]
                                    
                                    
                            cv2.putText(frame, str(id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 5)
        
                            
                            
            
                            if intersect(p0, p1, line4[0], line4[1]) and indexIDs[i] not in cross_check:
                                counter1 += 1
                                cross_check.append(indexIDs[i])
                                lane = 2 if ct1==1 else 3 if ct2==1 else 4 if ct3==1 else 5 if ct4==1 else 1
                                df2 = pd.DataFrame([[id,frameIndex,lane]], columns= ["TrackingID","FrameID","LaneID"])
                                df = df.append(df2,ignore_index=True)
                                
            #                if intersect(p0, p1, line2[0], line2[1]):
            #                    counter2 += 1
            
                        # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        #text = "{}".format(indexIDs[i])
                        #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #                frame = overlay(croped)
                i += 1
    
    
        # draw line
    #    cv2.line(frame, line1[0], line1[1], (255, 0, 255), 2)
    #    cv2.line(frame, line2[0], line2[1], (255, 0, 255), 2)
    #    cv2.line(frame, line3[0], line3[1], (255, 0, 255), 2)
        cv2.line(frame, line4[0], line4[1], (0, 255, 255), 3)
        ##############################################
        #FOR CALIBRATION
#        ##############################################
#        try:
#            min_time = min(time_for_speed)
#        except:
#            min_time = ("wait for it")
#        note_text = "current min time taken{}".format(min_time)
#        cv2.putText(frame, note_text, (50,110), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 6)
    
#        note_text = "NOTE: Vehicle speeds are calibrated only at yellow line. speed of cars are more stable."
#        cv2.putText(frame, note_text, (50,110), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
        # draw counter
        counter_text = "counter:{}".format(counter1)
        cv2.putText(frame, counter_text, (100,250), cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 0, 255), 7)
    #    cv2.putText(frame, "ctr2",str(counter2), (100,400), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 0, 255), 10)
        # counter += 1

    # saves image file
#    cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (1080, 720), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    new_dim = (1080,720)
    writer.write(cv2.resize(frame,new_dim, interpolation = cv2.INTER_AREA))

    # increase frame index
    frameIndex += 1

final_df1 = pd.merge_ordered(df, df4, how='outer', on='TrackingID')
final_df1 = final_df1[final_df1.FrameID.notnull()]
print("[INFO] cleaning up...")
writer.release()
vs.release()
final_df1.to_csv("/home/bamwani/final.csv",index=False)
