# Image Matching using CNN feature (Abstract)

## Overview
 
Unfortunately, the task of creating a model that can recognise satellite images in different weather conditions was not fully accomplished. 
From the results:
1) SIFT algorithm for determining the coincidence of two photos for the subsequent use of the obtained accuracy in the dataset.
2) Code structuring and pairing pictures with their matching accuracy.
3) Matrices with data for each picture for further implementation in the model.

Below are the steps to create the model and analysis of existing analogues on the analysis of constructions.

Main idea for preprocessing satellite photos: [Link](https://medium.datadriveninvestor.com/preparing-aerial-imagery-for-crop-classification-ce05d3601c68).
 
## SIFT Algorithm： 
<img alt="Image text" height="300" src="C:\Users\spark\OneDrive\Рабочий стол\sift1.jpg" width="500"/>

## Image Recognition Model (Buildings) 
<img alt="Image text" src="C:\Users\spark\OneDrive\Рабочий стол\123.jpg" width="500"/>

## Image Recognition Model (Buildings) 
<img alt="Image text" src="C:\Users\spark\OneDrive\Рабочий стол\555.jpg" width="500"/>

## SIFT Algorithm:
<img alt="Image text" src="C:\Users\spark\OneDrive\Рабочий стол\1231.jpg" width="500"/>

Comparing the results of the SIFT algorithm and the model trained on buildings one can notice the inconsistencies. But to make sure of it, I have uploaded satellite photos. The result is extremely negative.

## Getting start:
Python 3.7+ is recommended for running our code. 
1. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```

### Dataset
A dataset for the image matching task should include image pairs with associated associated associated labels indicating whether the images match or not. Each pair of images should have information about key points, descriptors or features highlighted in the images, which provides characterisation of objects and their environment.

### Prospectus for creating an effective model

1. **Using cool libraries:**
   - `tensorflow` or `keras` (to build a model and train it)
   - `numpy` (to work with data as arrays)
   - `matplotlib` (to show pictures)

2. **Preparing the data:**
   - We need pictures in pairs, where one pair matches the other (matches or doesn't match).

### Create Model

1. **Import cool layers and models**
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
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

3. **Let's give her the training data:**
```python
model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCHES, validation_data=(val_images, val_labels))
```
3. **Testing how good the model is**
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
predictions = model.predict(new_images)
```


 
