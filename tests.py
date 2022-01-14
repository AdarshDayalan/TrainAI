import unittest
import cv2
from body_recognition import process_img
import os

dir_path = os.path.dirname(os.path.realpath("Tests"))

class test_standing(unittest.TestCase):
    
    def test_standing(self):
        path = os.path.join(dir_path, "Tests\Standing\Images")
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path,filename))
            if img is not None:
                pose, feedback, counter = process_img(img)
                self.assertEqual(pose, "standing")

push_up_images = os.path.join(dir_path, "Tests\Push_Up\Images")
class test_push_up(unittest.TestCase):

    def test_push_up_0(self):
        img = cv2.imread(os.path.join(push_up_images,"pushup0.jpg"))
        pose, feedback, counter = process_img(img)
        self.assertEqual(pose, "push up")

    def test_push_up_1(self):
        img = cv2.imread(os.path.join(push_up_images,"pushup1.jpg"))
        pose, feedback, counter = process_img(img)
        self.assertEqual(pose, "push up")

    def test_push_up_2(self):
        img = cv2.imread(os.path.join(push_up_images,"pushup2.jpg"))
        pose, feedback, counter = process_img(img)
        self.assertEqual(pose, "push up")

    def test_push_up_3(self):
        img = cv2.imread(os.path.join(push_up_images,"pushup3.jpg"))
        pose, feedback, counter = process_img(img)
        self.assertEqual(pose, "push up")     

squat_images = os.path.join(dir_path, "Tests\Squat\Images")
class test_squat(unittest.TestCase):

    def test_squat_1(self):
        img = cv2.imread(os.path.join(squat_images,"squat1.jpg"))
        pose, feedback, counter = process_img(img)
        self.assertEqual(pose, "squat")

    def test_squat_2(self):
        img = cv2.imread(os.path.join(squat_images,"squat2.jpg"))
        pose, feedback, counter = process_img(img)
        self.assertEqual(pose, "squat")

    def test_squat_3(self):
        img = cv2.imread(os.path.join(squat_images,"squat3.jpg"))
        pose, feedback, counter = process_img(img)
        self.assertEqual(pose, "squat")

    def test_squat_4(self):
        img = cv2.imread(os.path.join(squat_images,"squat4.jpg"))
        pose, feedback, counter = process_img(img)
        self.assertEqual(pose, "squat")

sit_up_images = os.path.join(dir_path, "Tests\Sit_Up\Images")
class test_sit_up(unittest.TestCase):

    def test_sit_up_4(self):
        img = cv2.imread(os.path.join(sit_up_images,"situp4.jpg"))
        pose, feedback, counter = process_img(img)
        self.assertEqual(pose, "sit up")

if __name__ == '__main__':
    unittest.main()