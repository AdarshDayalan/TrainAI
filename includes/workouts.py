import imp
import cv2
import numpy as np
from numpy.lib.type_check import imag
from numpy.linalg import norm
import includes.constants as c

font = cv2.FONT_HERSHEY_SIMPLEX

def push_up_vis(vis):
    return (vis[12] > 0.8 and vis[14] > 0.8 and vis[16] > 0.8)

def squat_vis(vis):
    return (vis[24] > 0.8 and vis[26] > 0.8 and vis[28] > 0.8)

def situp_vis(vis):
    return (vis[12] > 0.8 and vis[24] > 0.8 and vis[26] > 0.8)
    
def push_up(image, lml):
    #Drawing Line from shoulder to elbow
    cv2.line(image, (lml[12][0], lml[12][1]),  (lml[14][0], lml[14][1]), (255, 0, 0), 2)
    #Drawing Line from elbow to wrist
    cv2.line(image, (lml[14][0], lml[14][1]), (lml[16][0], lml[16][1]), (0, 255, 0), 2)
    
    #Drawing line from nose to ankle
    nose_point = (lml[0][0], lml[0][1])
    ankl_point = (int((lml[27][0] + lml[28][0])/2) , int((lml[27][1] + lml[28][1])/2))
    hip_point = (int((lml[24][0] + lml[23][0])/2) , int((lml[24][1] + lml[23][1])/2))
    cv2.line(image, nose_point , ankl_point, (0, 0, 255) if c.feedback == "Keep hip inline with shoulders and legs" else (0,255,0), 2)

    #Calculates hip offset from straight back
    hip_error = perp_distance(nose_point, ankl_point, hip_point)
    if(hip_error > 20):
        c.feedback = "Keep hip inline with shoulders and legs"
    elif(c.feedback == "Keep hip inline with shoulders and legs"):
        c.feedback = "Hip Inline"
        
    cv2.putText(image, str(hip_error), hip_point, font, 0.9, (0,0,255), 2)

    #Calculates and outputs angle
    angle = get_angle(lml[12], lml[14], lml[16])
    cv2.putText(image, str(angle), (lml[14][0], lml[14][1]), font, 0.9, (0,0,255), 2)

    if angle < 100:
        c.push_up_down = True

    elif angle < 130:
        c.push_up_threshold = True

    elif angle > 150:
        if c.push_up_down:
            c.push_up_count += 1
            c.push_up_down = False
            c.feedback = "Good Pushup"
        elif c.push_up_threshold:
            c.feedback = "Pushup depth not reached"
        c.push_up_threshold = False

    # returns counter text
    counter = "Count: " + str(c.push_up_count)
    return counter

def squat(image, lml):
    feedback = ""
    counter = ""
    #get knee angle
    angle=get_angle(lml[24], lml[26], lml[28])
    #knee angle less than or equal to 90
    if angle <= 100:
        c.squat_down=True
        
    elif angle <= 140:
        c.squat_threshold = True
    #one rep completed after you have squatted down and come up to a knee angle of at least 150
    elif angle > 150:
        if c.squat_down:
            c.squat_count+=1
            c.squat_down=False
            c.feedback = "Good Squat"
        elif c.squat_threshold:
            c.feedback = "Squat Depth not reached"
        c.squat_threshold = False
    
    # midpoint between the two shoulders
    back_pt=(int(abs((lml[11][0] + lml[12][0])/2)), int(abs((lml[11][1] + lml[12][1])/2)))
    
    # add to np array if it has less than 20 elements
    if(len(c.motion_data) < 20):
        c.motion_data.append(back_pt)
    else:
        # delete first element
        c.motion_data.pop(0)
        # add new point
        c.motion_data.append(back_pt)

        for i in range(len(c.motion_data)):
            cv2.circle(image,c.motion_data[i],10,(0,255,0), 1)

        motion_x = [p[0] for p in c.motion_data]
        max_diff = np.amax(motion_x) - np.amin(motion_x)

        if(max_diff > 20):
            c.feedback = "Sqaut in a straighter line"
    
    # returns counter text
    counter = "Count: " + str(c.squat_count)
    return counter

def sit_up(image, lml):
    #Torso, shoulder, and knee angle (angle at hips)
    angle = get_angle(lml[12],lml[24],lml[26])
    cv2.putText(image, str(angle), (lml[12][0], lml[12][1]), font, 0.9, (0,0,255), 2)

    # at situp "up" position once hip angle <= 100
    if angle <= 100:
        c.situp_up = True
    # if hip angle <= 130, situp height not reached, bad form 
    elif angle <= 130:
        c.situp_threshold = True
    # back at situp "down" position when hip angle > 120 
    elif angle > 120:
        # if situp "up" was reached then increment rep count
        if c.situp_up:
            c.situp_count += 1
            c.situp_up = False
            c.feedback = "Good Situp"
        #  else if bad form displayed then no rep, display feedback
        elif c.situp_threshold:
            c.feedback = "Situp height not reached"
        # situp_threshold (form check) reset to False after reaching situp "down" position,
        c.situp_threshold = False       
    
    # returns counter text
    text = "Count: " + str(c.situp_count)
    return text

#Helpers 
# return angle (degrees) given three lml points
def get_angle(pt1, pt2, pt3):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    pt3 = np.array(pt3)
    
    ba = pt1 - pt2
    bc = pt3 - pt2

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return int(np.degrees(angle))
# get 
def perp_distance(pt1, pt2, pt3):
    p1 = np.array(pt1)
    p2 = np.array(pt2)
    p3 = np.array(pt3)

    d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    return int(d)
