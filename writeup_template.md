# **Behavioral Cloning** 

[//]: # (Image References)
[image1]: ./output/NVidia_model.jpg "NVidia Model Visualization"
[image2]: ./output/recoveryside.jpg "Recovery from side of road"
[image3]: ./output/recovery_dirt.jpg "Recovery from dirt"
[image4]: ./output/centerdriving.jpg "Center lane driving"
[video1]: ./output/output_video.mp4 "Track 1 video"
[video2]: ./output/output_jungle.mp4 "Jungle track video"


The goal of this project is to build a convolutional network in Keras that can autonomously steer a car around a a simulated track.  As you can see in the two output videos, the model sucessfully navigates the car around both the simple and the advanced tracks.

Simple track:
![alt text][video1]


Jungle track:
![alt text][video1]



## Sources
My work for this project relies heavily on two sources:
* David Silver's lecture notes
* [The Nvidia self driving car model architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

In addition, I learned tips from:
•	ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst
•	SSD: Single Shot MultiBox Detector
•	Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net 
•	SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation 
•	https://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* output video of a car driving autonomously around the first track
![alt text][video1]

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the NVidia model for my project (model.py lines 98-111). Details described below. 

#### 2. Attempts to reduce overfitting in the model

Using the model as-is with 25k image points, I found no overfitting.  Validation and Training accuracy were acceptable.  Likewise, the car drove autonomously without problem.  I played with dropout and L2 normalization, but found them to be unnecessary.  

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 118).

#### 4. Appropriate training data

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


The overall strategy for deriving a model architecture was to use a convolutional network on the image data leading to a single output.  I used adam optimizer to minimize mean square error of the predicted steering angle.

After considering a number of model architectures, I decided to use NVidia's proven solution.  The model performed well immediately, neither overfitting nor underfitting.  Training and validation accuracy were approximately 95%.  Likewise, when driven autonomously, the car made it nearly fully around the track.  As will be described below, I built the training set to address the few points where the car fell off the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

I used the NVidia model for my project (model.py lines 98-111).  It has 5 convolutional layers, some using 5x5 filters and some using 3x3 filters.  This is followed by 3 fully connected layers and then the output.  All activations are RELU.
![alt text][image1] 


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to reamin in the center of the lane. These images show what a recovery looks like:

![alt text][image2]

Then I repeated this process on track two in order to get more data points.

However, I still found that the car tended to leave the road when it encountered a dirt shoulder.  To train against this behavior, I speficially recorded training data showing the car returning to the center lane from dirt shoulders.  It looks like this:

![alt text][image3]


All turns in track 1 are to the left.  The model is likely to learn that hugging the left lane marker is appropriate behavior.  To combat this instinct, I had two options:  drive around the track in the other direction or to flip images and angles.  I chose the second.  It is implemented as part of the generator code in lines 73-77 of model.py.  

After the collection process, I had X number of data points. I  randomly shuffled the data set and put 20% of the data into a validation set using sklearn's train_test_split. 

I placed preprocessing steps in my data generator (lines 50-83 of model.py).  I dind't need very much.  I ensured the images were in RGB so drive.py could work well, and I flipped images as described earlier.

I trained for 5 epochs, at which point accuracy ceased to improve.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
