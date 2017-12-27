## Project: Semantic Segmentation

---

[//]: # (Image References)

[image1]: ./learning_curve.png
[image2]: ./mean_iou_curve.png

## [Rubric](https://review.udacity.com/#!/rubrics/989/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Build the Neural Network

* Please see `main.py`.

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

* On average, the model decreases loss over time.
  * The following diagram shows the learning curve of my model. 

![alt text][image1]

#### 2. Does the project use reasonable hyperparameters?

* The number of epoch and batch size are set to a reasonable number.
  * I tested 16, 32 for the batch size.
    * I found that optimization process is done faster under 16 than 32. Finally I fixed it as 16.
  * I tested 175 epoches. This value is written in the original FCN paper.
    * I was able to get more than 82% mean IOU for testing data after 165 epoches.
  * I set Learning rete as 1e-3. This is very usual setup.

#### 3. Does the project correctly label the road?

* The project labels most pixels of roads close to the best solution. The model doesn't have to predict correctly all the images, just most of them.
* A solution that is close to best would label at least 80% of the road and label no more than 20% of non-road pixels as road.

  * My model got 0.830137 mean iou for testing data. Unfortunately there is no ground truth labels for validation or testing. And testing set is too small to split into testing data and validation data. But the final testing result is so reasonable to see as follows.
![alt text][image2]

<div style="position:relative;height:0;padding-bottom:56.25%"><iframe src="https://www.youtube.com/embed/ytjy5uhnnAw?ecver=2" width="640" height="360" frameborder="0" gesture="media" allow="encrypted-media" style="position:absolute;width:100%;height:100%;left:0" allowfullscreen></iframe></div>

