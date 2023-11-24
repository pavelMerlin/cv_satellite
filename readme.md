# Image Matching using CNN feature (Abstract)

## Overview
 
The task of creating a model that can recognise satellite images in different weather conditions was not fully accomplished. 
From the results:
1) SIFT algorithm was used as a baseline benchmark for key points detection and provision of image matching accuracy (`gaussian_algorinm.ipynb`).
2) Pretrained model open for finetuning on stellite images was not found in given amount of time.
3) Several deep learning models were found and applied to key point detection and image matching task. Unfortunately, they only provided inference examples; both architecture and traning remained hidden.
4) Training classical ML models or CNN arcitecture models would be prefered if more time was provided.

Below are the steps to create the model and analysis of existing analogues on the analysis of constructions.

Main idea for preprocessing satellite photos: [Link](https://medium.datadriveninvestor.com/preparing-aerial-imagery-for-crop-classification-ce05d3601c68).

A model for recognising differences in buildings [Link](https://www.kaggle.com/code/cbeaud/imc-2022-kornia-score-0-725/notebook).

### Prerequisites

Python 3.7+ is recommended for running our code. 
1. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```

## Examples

In `preprocess.ipynb` we can find benchmark code solution for key points detection and Dataset creation.

### SIFT Algorithm： 
<p align="center">
 <img width="700px" src="https://github.com/pavelMerlin/cv_satellite/blob/master/train/results/sift1.jpg" alt="qr"/>
</p>

### Image Recognition Model (Buildings) 
<p align="center">
 <img width="700px" src="https://github.com/pavelMerlin/cv_satellite/blob/master/train/results/match3.jpg" alt="qr"/>
</p>

### Image Recognition Model (Buildings) 
<p align="center">
 <img width="700px" src="https://github.com/pavelMerlin/cv_satellite/blob/master/train/results/match4.jpg" alt="qr"/>
</p>

### SIFT Algorithm:
<p align="center">
 <img width="700px" src="https://github.com/pavelMerlin/cv_satellite/blob/master/train/results/sift2.jpg" alt="qr"/>
</p>

### Image Recognition Model (Buildings) Good Example
<p align="center">
 <img width="700px" src="https://github.com/pavelMerlin/cv_satellite/blob/master/train/results/match1.jpg" alt="qr"/>
</p>

Comparing the results of the SIFT algorithm and the model trained on buildings, you can see inconsistencies when not working with buildings. But to make sure of it, I uploaded the pictures. The result is extremely negative. 

Unfortunately, I could not find a train-ready model architecture for my task.

## Next steps

### Dataset preparation

A dataset for the image matching task should include image pairs with associated associated associated labels indicating whether the images match or not. Each pair of images should have information about key points, descriptors or features highlighted in the images, which provides characterisation of objects and their environment.

Benchmarking code mentioned above is synced with this dataset and can be re-used for future models.

### Prospectus for creating an effective model

1. **Using cool libraries:**
   - `tensorflow` or `keras` (to build a model and train it)
   - `numpy` (to work with data as arrays)
   - `matplotlib` (to show pictures)

2. **Preparing the data:**
   - We need pictures in pairs, where one pair matches the other (matches or doesn't match).

### Create Model

1. **Import cool layers and models**
This structure has nine layers of neurons connected in series. This model is well suited for solving the clustering problem.
```python
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(высота_картинки, ширина_картинки, каналы)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

2. **Let's get the model ready to go:**
We choose the adam optimizer, binary crossentropy shows excellent results when paired with optimizer. Show metrics with loss function and accuracy values.
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

3. **Let's give her the training data:**
```python
model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCHES, validation_data=(val_images, val_labels))
```

4. **Testing how good the model is**
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
predictions = model.predict(new_images)
```
