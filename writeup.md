## Project: Semantic Segmentation

---

[//]: # (Image References)

[image1]: ./learning_curve.png
[image2]: ./mean_iou_curve.png

## [Rubric](https://review.udacity.com/#!/rubrics/989/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Build the Neural Network

#### 1. Does the project load the pretrained vgg model?

The function load_vgg is implemented correctly.

#### 2. Does the project learn the correct features from the images?

The function layers is implemented correctly.

#### 3. Does the project optimize the neural network?

The function optimize is implemented correctly.

#### 4. Does the project train the neural network?

The function train_nn is implemented correctly. The loss of the network should be printed while the network is training.

### Neural Network Training

#### 1. Does the project train the model correctly?

On average, the model decreases loss over time.

![alt text][image1]

#### 2. Does the project use reasonable hyperparameters?

The number of epoch and batch size are set to a reasonable number.

#### 3. Does the project correctly label the road?

The project labels most pixels of roads close to the best solution. The model doesn't have to predict correctly all the images, just most of them.

A solution that is close to best would label at least 80% of the road and label no more than 20% of non-road pixels as road.

![alt text][image2]

<iframe width="560" height="315" src="https://www.youtube.com/embed/wT3g86IPgm0" frameborder="0" gesture="media" allow="encrypted-media" allowfullscreen></iframe>

