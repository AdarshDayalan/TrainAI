import cv2
import csv
import mediapipe as mp
import numpy as np
import time
import itertools
import copy
import os

from numpy.core.defchararray import count
from includes.body_model import probability_model
import includes.constants as c
from includes.workouts import push_up, push_up_vis, squat, squat_vis, sit_up, situp_vis

#sets up constant mediapipe variables
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        enable_segmentation=True,
        smooth_landmarks=True)

#Defining global variables
csv_path = "CSV/body_landmarks.csv"
# csv_path = "C:/Users/Basilio/Documents/yes we can project/final-project-yeswecan/CSV/body_landmarks.csv"
logging = False
logging_num = 4

pose = "None"

prev_feedback = c.feedback
feedback_changed = False

start_t = time.time()

#Logs landmark list to CSV file
def logging_csv(number, landmark_list):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])

#Calculates landmark list to xy coordinates for the image
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
 
    landmark_point = []
    visibility_point = []
 
    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_visibility = landmark.visibility
 
        landmark_point.append([landmark_x, landmark_y, landmark_z])
        visibility_point.append(landmark_visibility)
 
    return landmark_point, visibility_point

#Normalizes data so AI can train efficiently
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
 
    # Convert landmark to relative coordinates
    base_x, base_y = 0,0 
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]
 
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z
 
    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
 
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
 
    def normalize_(n):
        return n / max_value
 
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
 
    return temp_landmark_list

def process_img(image):
    image = cv2.flip(image, 1)

    # image = cv2.resize(image, (600,700), interpolation= cv2.INTER_LINEAR)

    height, width, channels = image.shape
    black_img = np.zeros((height, width, channels), dtype = "uint8")

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #Draws landmark to black image
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

    counter = ""

    #If body is detected
    if(results.pose_landmarks):
        lml, vis = calc_landmark_list(image, results.pose_landmarks)
        pre_lml = pre_process_landmark(lml)

        if(logging):
            logging_csv(logging_num, pre_lml)

        else:
            #Converts to np array to be put in neural network
            pre_lml = (np.expand_dims(pre_lml,0))
            cv2.imwrite("body.jpg", image)
            global pose
            #Probability results of neural network on each body position
            probability_values = probability_model.predict(pre_lml)[0]
            if(np.max(probability_values) > 0.8):
                c.current_state = np.argmax(probability_values)
                if(c.current_state == 1 and push_up_vis(vis)):
                    counter = push_up(image, lml)
                elif(c.current_state == 3 and squat_vis(vis)):
                    counter = squat(image, lml)
                elif(c.current_state == 4 and situp_vis(vis)):
                    counter = sit_up(image,lml)
                pose = c.label_names[c.current_state]

    #Removes feedback after 5 seconds
    global prev_feedback
    global feedback_changed
    global start_t
    
    if(prev_feedback != c.feedback):
        feedback_changed = True
        start_t = time.time()
        prev_feedback = c.feedback

    if(feedback_changed):
        time_df = time.time() - start_t
        if(time_df > 5):
            c.feedback = ""
            feedback_changed = False
    
    #Outputs counter, feedback, and pose classification
    counter_size = cv2.getTextSize(counter, c.font, 2, 4)[0]
    counter_pnt = (int(width - counter_size[0]), counter_size[1] + 5)

    feedback_size = cv2.getTextSize(c.feedback, c.font, 0.9, 2)[0]
    feedback_pnt = (int(width/2 - feedback_size[0]/2), height - feedback_size[1] - 5)

    cv2.putText(image, counter, counter_pnt, c.font, 2, (0, 0, 255), 4)
    cv2.putText(image, c.feedback, feedback_pnt, c.font, 0.9, (255, 0, 0), 2) 
    cv2.putText(image, pose, (10, 20), c.font, 0.9, (0,0,255), 2)  

    cv2.imshow('Feed', image)
    cv2.imshow('MediaPipe Holistic', black_img)

    return pose, c.feedback, counter

def main():
    cap = cv2.imread("pushup.jpg")
    process_img(cap)

    while cap.isOpened():
        success, image = cap.read()
        process_img(image)
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == '__main__':
    main()
    # dir_path = os.path.dirname(os.path.realpath("Tests"))
    # push_up_images = os.path.join(dir_path, "Tests\Push_Up\Images")

    # img = cv2.imread(os.path.join(push_up_images,"pushup4.jpg"))
    # pose, feedback, counter, img = process_img(img)
    # while True:
    #     cv2.imshow('Output', img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break