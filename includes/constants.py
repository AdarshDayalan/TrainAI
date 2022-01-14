import cv2

label_names = ['standing', 'push up', 'sitting', 'squat', 'sit up']

font = cv2.FONT_HERSHEY_SIMPLEX

current_state = 0
prev_state = 0

push_up_down = False
push_up_count = 0
push_up_threshold = False

motion_data= []
squat_down = False
squat_count = 0
squat_threshold = False

situp_up = False
situp_count = 0
situp_threshold = False

feedback = ""
feedback_changed = False