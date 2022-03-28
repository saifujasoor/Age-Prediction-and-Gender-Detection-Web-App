# Age-Prediction-and-Gender-Detection
A gender Detection  and age prediction web application developed using Flask, Open CV, HTML & CSS
### Built With
![python-shield] ![flask] ![open-cv] ![html-shield] ![css-shield] 

* The frontend of the application is built using HTML & CSS
* The backend is built using Flask and Open CV

### Age Prediction and Gender Detection

First introducing you with the terminologies used in this advanced python project of gender and age detection –

### What is Computer Vision?
Computer Vision is the field of study that enables computers to see and identify digital images and videos as a human would. The challenges it faces largely follow from the limited understanding of biological vision. Computer Vision involves acquiring, processing, analyzing, and understanding digital images to extract high-dimensional data from the real world in order to generate symbolic or numerical information which can then be used to make decisions. The process often includes practices like object recognition, video tracking, motion estimation, and image restoration.

### What is OpenCV?
OpenCV is short for Open Source Computer Vision. Intuitively by the name, it is an open-source Computer Vision and Machine Learning library. This library is capable of processing real-time image and video while also boasting analytical capabilities. It supports the Deep Learning frameworks TensorFlow, Caffe, and PyTorch.

### What is a CNN?
A Convolutional Neural Network is a deep neural network (DNN) widely used for the purposes of image recognition and processing and NLP. Also known as a ConvNet, a CNN has input and output layers, and multiple hidden layers, many of which are convolutional. In a way, CNNs are regularized multilayer perceptrons.

### Gender and Age Detection Python Project- Objective
To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture using Deep Learning on the Adience dataset.

### Gender and Age Detection – About the Project
In this Python Project, we will use Deep Learning to accurately identify the gender and age of a person from a single image of a face. We will use the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, we make this a classification problem instead of making it one of regression.

Ref: https://data-flair.training/blogs/python-project-gender-age-detection/

## Getting Started

### Prerequisites
Make sure you have Python, OpenCV, Flask and PIL installed on your system to run this project.

### Execution guide
1. Download the contents of the repository
2. Make sure the necessary prerequisites are installed on your system
3. Type the following command inside the directory on your terminal
  ```sh
  python3 app.py
  ```
4. Click http://127.0.0.1:5000/ (Press CTRL+C to quit)

## Project Demo
### Project Overview

![s1]

### Live detection

![s2]

### Detection using image

![s3]

## Contact
[![linkedin-shield]][linkedin]


<!-- Links -->

[python-shield]: https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white&style=for-the-badge
[open-cv]: https://img.shields.io/badge/-OpenCV-red?logo=opencv&logoColor=white&style=for-the-badge
[flask]: https://img.shields.io/badge/-Flask-black?logo=flask&logoColor=white&style=for-the-badge
[html-shield]: https://img.shields.io/badge/-HTML-orange?logo=html5&logoColor=white&style=for-the-badge
[linkedin-shield]: https://img.shields.io/badge/-linkedin-0078B6?logo=linkedin&logoColor=white&style=for-the-badge
[linkedin]:https://www.linkedin.com/in/saifullahrahimi/
[1]: ./Demo/s1.gif
[2]: ./Demo/s2.gif
[3]: ./Demo/s3.gif
