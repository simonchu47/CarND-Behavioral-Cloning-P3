#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/loss_compare.png "Loss Compare"
[image3]: ./examples/center_2017_07_20_12_44_59_857.jpg "Recovery Image"
[image4]: ./examples/center_2017_07_19_15_24_29_651.jpg "Recovery Image"
[image5]: ./examples/center_2017_07_19_15_24_04_345.jpg "Recovery Image"
[image6]: ./examples/center_2017_07_19_15_35_22_859.jpg "Recovery Image"
[image7]: ./examples/center_2017_07_19_15_39_20_985.jpg "Recovery Image"
[image8]: ./examples/loss_compare_2.png "Loss Compare"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network similar to NVIDIA's one, including the following layers.

Layer 1, a convolutional feature mapping with a 5x5 filter size and depth 24, following with max pooling 2x2 and RELU activation.

Layer 2, a convolutional feature mapping with a 5x5 filter size and depth 36, following with max pooling 2x2 and RELU activation.

Layer 3, a convolutional feature mapping with a 5x5 filter size and depth 48, following with max pooling 2x2 and RELU activation.

Layer 4, a convolutional feature mapping with a 3x3 filter size and depth 64, following with RELU activation.

Layer 5, a convolutional feature mapping with a 3x3 filter size and depth 64, following with RELU activation and is then flatten.

Layer 6, a fully-connected layer to 100 neurons.

Layer 7, a fully-connected layer to 50 neurons.

Layer 8, a fully-connected layer to 10 neurons.

Layer 9, a fully-connected layer to 1 output.

Before the structure, fed-in images are first cropped(top 50 pixels and bottom 20 pixels with original resolution 160x320) and then normalized.

####2. Attempts to reduce overfitting in the model

This model was first trained with only one lap recorded images of track 1(the left one). These data are splited into training and validation data sets to avoid the model was not overfitting. But after 5 epochs training, the model was still becoming overfitting.

The model was tested by running it through the simulator, but the vehicle could not stay on the track 1 or track2.

I decided to enlarge the training data including more and more recorded images of laps on both track 1 and track 2(the right one).


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 131).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. On the simulator, I found that the vehicle would have understeering at some specific turns.

I used a combination of center lane driving, recovering from the left and right sides of the road. Especially, for those turns at which understeering happened, the tracks of recovering from the side to prevent bumping over the curb are also added into the training data.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to refer the experiences of NVIDIA described on the paper. And small amounts of training data was used to make sure the happening of overfitting which showed the capacity of the model would be enough.

My first step was to use a convolution neural network model similar to NVIDIA's cnn architecture. I thought this model might be appropriate because it was proved on a real road.

In order to gauge how well the model was working, first I trained the model with recorded data of only one lap on track 1, including images and steering angle data, which are splited into a training and a validation set. I found that after 5 epochs this model had overfitting as shown below. Tested on the simulator, the model could not keep the vehicle on the road.

![alt text][image2]

To combat the overfitting, I enlarged the size of the training data, which were including data of more laps on both track 1 and track 2. This time the vehicle could keep on the road of track 1. But there would be understeering at some specific turns.

Then I decided to use transfer learning. I kept the model's weights of last time, and tried to re-train the model with some data of recovering from sides at some turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture was the same as the originally designed one, which has 9 layers. Here is a visualization of the architecture.

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery from both sides back to center. These images show what a recovery looks like starting from the understeering or oversteering when passing through some turns.

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

![alt text][image6]
![alt text][image7]

After the collection process, I had 23586 number of data points. I then preprocessed the images of this data by cropping the unnecessary portion of view, which were 50 pixels from the top and 20 pixels from the bottom. Then the images were normalized.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the loss plot of training and validation, which showed that too many epochs might cause the model overfitting.

![alt text][image8]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
