# final-project-yeswecan
A workout motion tracker. 

Collaborators: Adarsh Dayalan [dayalan2], Samuel Du [sydu2] , Zhengru (James) Fang [zhengru3], Basilio Leopoldo [bleopo2]

## Description
In this project, we combine a body landmark tracker with machine learning in order to track workout motions. The body landmark tracker calculates the landmarks on the human body on an image or a video, while the machine learning classifies whether the person in the video is standing, sitting, performing push-ups or squats. 

We use Google's MediaPipe library in order to track the landmarks on the human body. The MediaPipe library has collected millions of data points and serves this purpose very well. However, detecting the exercise state of the person in the video would still require the use of further classification using machine learning. To track the performance of the exercise, some hard encoding of the "geometries" derived from the tracked landmarks was used. 

![image](https://user-images.githubusercontent.com/42521639/144725591-759739f3-9eeb-46ac-8b33-af31ec7ae792.png)

## Required Libraries
Below is a list of required libraries that you will need to have installed:

- Tensorflow
- CV2
- Numpy 
- MediaPipe
- Pandas

## Manual
To train the model from the provided CSV file:

```
python3 includes/train_model.py
```

To use the application with the trained model:

```
python3 body_recognition.py
```

To run the testcases:

```
python3 test.py
```

## Sample Results
Pushup Demo:

https://user-images.githubusercontent.com/46902243/145101707-6a97414d-e6f5-4040-afe7-87183dd2869d.mp4

Squat Demo:

https://user-images.githubusercontent.com/46902243/145101738-0be123d5-81d2-4e70-9d50-54d1148beda7.mp4

Situp Demo:

https://user-images.githubusercontent.com/53020083/145107560-8ee1bf71-3719-4ac4-a2a7-40a0bb093ee7.mp4


## References and Notes
The model was trained with 99, 132, 33 dense layers in between each training iteration. 
