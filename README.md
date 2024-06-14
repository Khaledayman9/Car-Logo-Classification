# Car-Logo-Classification
A CNN model that can accurately identify which car brand a logo belongs to among eight possible brands: Hyundai, Lexus, Mazda, Mercedes, Opel, Skoda, Toyota, and Volkswagen. This classification task is a subset of image recognition, a fundamental problem in computer vision.


# 1. Problem:
The problem at hand is the classification of car brand logos using a Convolutional Neural Network (CNN). The task involves training a model to identify which car brand a logo belongs to among eight possible brands: Hyundai, Lexus, Mazda, Mercedes, Opel, Skoda, Toyota, and Volkswagen. This classification task is a subset of image recognition, a well-known problem in computer vision.

# 2. Aim:
The primary aim of this task is to develop an accurate and robust model capable of classifying car brand logos from images. Achieving this goal can have practical applications in various fields, including automated quality control in manufacturing, visual marketing, and brand monitoring on social media.

# 3. Importance:
1.	Automation: Automated logo classification can significantly reduce the time and effort required for manual identification and sorting of images.
   
2.	Scalability: A reliable model can handle large datasets, making it feasible to analyze extensive collections of images, which is impractical manually.
   
3.	Precision Marketing: Knowing where and how often brand logos appear can help companies tailor their marketing strategies.
   
4.	Intellectual Property Protection: Detecting unauthorized use of logos can help in protecting brand identity and intellectual property rights.

# 4. Challenges:

1.	Variability in Logos: Logos might vary in color, size, orientation, and context in which they appear, making consistent classification challenging.
   
2.	Data Imbalance: Some brands may have more images available than others, leading to an imbalanced dataset that can bias the model.
   
3.	Overfitting: Given the limited number of classes and potential for many similar features among logos, the model might overfit to the training data.

4.	Generalization: Ensuring that the model performs well on unseen data (generalization) is crucial but challenging due to possible variations not represented in the training set.




# 5. Methodology

**1. Data Preparation:**

   - **Dataset Splitting**: The dataset was split into training (80%), validation (20%), and test sets to ensure the model is evaluated on unseen data.
   
   - **Data Augmentation**: Techniques such as rotation, width and height shifts, shearing, zooming, and horizontal flipping were applied to artificially increase the diversity of the training data, helping the model generalize better.
      
**2. Model Architecture:**

   - A Sequential CNN model was constructed with the following characteristics:
      
      1. **Conv2D Layers**: Multiple convolutional layers with swish activation functions and HeNormal initializers were used to extract features from the input images.
          
      2. **MaxPooling2D**: Pooling layers were introduced to reduce spatial dimensions and computational load.
          
      3. **BatchNormalization**: Applied after pooling and dense layers to stabilize and accelerate training.
          
      4. **Dropout**: Used to prevent overfitting by randomly dropping units during training.
          
      5. **Dense Layers**: Fully connected layers were used towards the end of the network for classification, with the final layer having a softmax activation function to output probabilities for each class.
  
**3. Training and Evaluation:**

   - **Optimizer and Loss Function**: The model was compiled using the Adam optimizer and categorical crossentropy loss, suitable for multi-class classification.
      
   - **Training**: The model was trained for 200 epochs, with training performance monitored using accuracy and loss metrics.
      
   - **Evaluation**: Post-training, the model was evaluated on the test set, and metrics such as accuracy, precision, recall, and F1-score were calculated.


# 6. Dataset:
The selection of the Kaggle dataset, "Car Brand Logos" by Volkandl, for this case study was driven by several key factors that align well with the objectives and requirements of developing a CNN model for car brand logo classification. To access dataset[^1]. Below, I outline the reasoning behind this choice based on the provided code and the dataset characteristics. 

[^1]: [Car Brand Logos](https://www.kaggle.com/datasets/volkandl/car-brand-logos)

1. Dataset Relevance and Specificity
The primary goal of this project is to classify car brand logos. The Kaggle dataset specifically focuses on car brand logos, which makes it highly relevant. It contains images labeled by brand, providing the necessary annotations to train and evaluate a supervised learning model effectively.

2. Diverse Set of Brands
The dataset includes logos from eight distinct car brands: Hyundai, Lexus, Mazda, Mercedes, Opel, Skoda, Toyota, and Volkswagen. This diversity is essential to train a robust model that can differentiate between multiple classes, mimicking real-world scenarios where a variety of brands need to be recognized.

3. Adequate Data Quantity
With over 2000 images in the training set and a balanced number of images per class, the dataset offers a reasonable amount of data for training a CNN. This quantity helps in achieving a good representation of each brand, which is crucial for the model to learn the distinguishing features of each logo.

4. Train, Validation, and Test Split
The dataset structure allows for easy splitting into training, validation, and test sets. This is evident in the code where an 80-20 split is used for training and validation, and a separate test set is used for final evaluation. Such a structure is ideal for building and assessing machine learning models, ensuring that performance metrics are evaluated on unseen data.

5. Pre-labeled and Organized
The dataset is pre-labeled and organized into directories based on brand names. This organization simplifies the process of using data generators like ImageDataGenerator in Keras, which can automatically label the data based on directory names. This reduces the preprocessing effort and potential for human error in manual labeling.

6. High-Quality Images
The images in the dataset are of high quality, which is important for training CNNs that often require clear and well-defined input data to extract meaningful features. High-quality images help in improving the accuracy and generalization capability of the model.

7. Availability and Accessibility
Kaggle is a well-known platform that provides easy access to datasets for machine learning projects. The "Car Brand Logos" dataset is freely available, making it accessible to researchers and practitioners. This ease of access facilitates reproducibility and further experimentation.

8. Community and Support
Using a dataset from Kaggle provides the additional benefit of community support. Kaggle datasets often come with discussions, kernels (notebooks), and other resources that can help in understanding and working with the data more effectively.


# 7. Attributes and Hyperparameters:
In the context of the CNN model for car brand logo classification, the attributes chosen for the model and those to be predicted play a crucial role in determining the model's effectiveness. Here's a detailed discussion on these attributes:

### 1. Attributes Chosen for the Predictive Model

The primary attributes chosen for the predictive model are the image features extracted through the convolutional layers of the CNN. Here's why each chosen attribute and preprocessing step is important:

#### Image Rescaling (rescale=1./255):

- **Importance**: Normalizing pixel values to the range [0, 1] is essential for improving the model's convergence during training. This scaling helps in stabilizing the gradient descent optimization process.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
```
#### 	Data Augmentation:

- **Importance**: Techniques such as rotation, width and height shifts, shear, zoom, and horizontal flipping increase the diversity of the training dataset. This helps the model generalize better by learning to recognize logos in various orientations and transformations, simulating real-world scenarios where logos might not always appear in a standard format. For example, in the code snippet, it can be seen that the rotation_range is 20, width_shift_range = 0.1 and so on.
  
#### 	Convolutional Layers (Conv2D):

- **Importance**: These layers are responsible for feature extraction. They detect local patterns such as edges, textures, and shapes that are essential for distinguishing different car logos. The swish activation function and HeNormal initializer which is a method used to initialize the weights of neural networks are chosen to improve learning efficiency and model performance.The kernel size was 3x3 for all convolutional layers.Same padding was used also in all convolutional layers.

```python
model = models.Sequential([
   layers.Conv2D(32, (3, 3), activation='swish', padding='same', input_shape=(224, 224, 3), kernel_initializer=HeNormal()),
   layers.Conv2D(32, (3, 3), activation='swish', padding='same', kernel_initializer=HeNormal()),
   layers.MaxPooling2D((2, 2)),    layers.BatchNormalization(),
   layers.Dropout(0.3),    ...
])
```

#### 	Pooling Layers (MaxPooling2D):
- **Importance**: These layers reduce the spatial dimensions of the feature maps, thereby reducing the computational complexity and helping in retaining the most important features. They also help in achieving spatial invariance, which is crucial for recognizing logos irrespective of their position in the image. The kernel size for the pooling layers was 2x2 and Max pooling was used in all Pooling layers in the CNN.

```python
layers.MaxPooling2D((2, 2)),
```


#### Batch Normalization:
- **Importance**: This helps in stabilizing and speeding up the training process by normalizing the input to each layer. It mitigates issues like internal covariate shift and makes the model more robust.

```python
layers.BatchNormalization(),
```

#### Dropout:
- **Importance**: Dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to zero during training. This encourages the network to learn redundant representations and enhances generalization. We used in the implementation multiple Dropout layers with different dropout factors (ex: 0.3, 0.4,and 0.5) to address overfitting.

```python
layers.Dropout(0.3),
```

#### Dense Layers:
- **Importance**: Fully connected layers towards the end of the network consolidate the features extracted by the convolutional layers to make final predictions. The swish activation function in these layers ensures smooth and non-linear transformations, aiding in better learning.

```python
layers.Flatten(),
layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(0.001), kernel_initializer=HeNormal()),
layers.Dropout(0.5),
layers.BatchNormalization(),
layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(0.001), kernel_initializer=HeNormal()),
layers.Dropout(0.5),
layers.BatchNormalization(),
```

#### Softmax Activation in Output Layer:
- **Importance**: The softmax function is used in the output layer to convert the logits into probabilities for each class (car brand). This helps in interpreting the network's predictions as probabilities, facilitating classification.

```python
layers.Dense(8, activation='softmax')
```	    

#### Adam Optimizer:
- **Importance**: The Adam (Adaptive Moment Estimation) optimizer is crucial for efficiently training deep learning models. It combines the advantages of two other popular optimizers: AdaGrad and RMSProp. Adam is particularly useful for models that involve large datasets and high-dimensional parameter spaces, such as those used for image classification tasks like recognizing car brand logos. A learning rate of 0.001 was chosen and the loss function is “Categorical Cross Entropy” for multiclass classification and the metrics is accuracy. The model was trained for 200 epochs. The batch size for was 64.

```python
model.compile(optimizer= Adam(learning_rate=0.001),
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
```	    


### 2. Attributes to be Predict
The attributes to be predicted are the car brand labels corresponding to the logos. In this case, the labels are:
1.	Hyundai
2.	Lexus
3.	Mazda
4.	Mercedes
5.	Opel
6.	Skoda
7.	Toyota
8.	Volkswagen
The model aims to classify each input image into one of these eight classes.
#### Importance of the Predicted Attributes:
-	Brand Recognition: Accurate prediction of car brand logos has direct implications for brand recognition and monitoring. It helps in ensuring that the brand identity is correctly identified and tracked across various media.
-	Market Analysis: By identifying the presence and frequency of different car brands in images, companies can gain insights into market trends and consumer preferences.
-	Intellectual Property Protection: Identifying and classifying logos correctly can aid in detecting unauthorized use of trademarks and logos, helping in protecting intellectual property.
-	Enhanced User Experience: Applications such as automated sorting and searching of images based on car brands can greatly enhance user experience in digital asset management systems.


# 8. Architecture:
In developing a CNN model for car brand logo classification, the chosen architecture and parameters are crucial for achieving optimal performance. The model architecture is designed to effectively extract and learn the features of car brand logos, facilitating accurate classification. Here are the key components and their rationale:

## Model Architecture:


###  Sequential Model:

- **Reasoning**: A Sequential model is used to build the network layer-by-layer in a straightforward manner, which is suitable for image classification tasks.
```python
model = models.Sequential([
...
])
```

### Convolutional Layers (Conv2D):

- **Reasoning**: Convolutional layers are the backbone of CNNs. They are used to automatically and adaptively learn spatial hierarchies of features from input images. The layers use filters to convolve with the input image, capturing essential patterns such as edges, textures, and shapes.

```python
model = models.Sequential([
   layers.Conv2D(32, (3, 3), activation='swish', padding='same', input_shape=(224, 224, 3), kernel_initializer=HeNormal()),
   layers.Conv2D(32, (3, 3), activation='swish', padding='same', kernel_initializer=HeNormal()),
   layers.MaxPooling2D((2, 2)),    layers.BatchNormalization(),
   layers.Dropout(0.3),    ...
])
```


### Activation Function (Swish):

- **Reasoning**: The swish activation function, defined as \[ f(x) = x \cdot \sigma(x) \] where \( \sigma(x) = \frac{1}{1 + e^{-x}} \) is the sigmoid function., has been shown to perform better than ReLU in deep networks due to its smooth and non-monotonic properties. This helps in achieving better convergence and performance.

```python
model = models.Sequential([
   layers.Conv2D(32, (3, 3), activation='swish', padding='same', input_shape=(224, 224, 3), kernel_initializer=HeNormal()),
   layers.Conv2D(32, (3, 3), activation='swish', padding='same', kernel_initializer=HeNormal()),
])
```

### Kernel Initializer (HeNormal):

- **Reasoning**: The HeNormal initializer is used to set the initial random weights of the network. It helps in maintaining the variance of the weights through layers, which is crucial for effective training of deep networks.

```python
model = models.Sequential([
   layers.Conv2D(32, (3, 3), activation='swish', padding='same', input_shape=(224, 224, 3), kernel_initializer=HeNormal()),
   layers.Conv2D(32, (3, 3), activation='swish', padding='same', kernel_initializer=HeNormal()),
])
```

### Pooling Layers (MaxPooling2D):
- **Reasoning**: Pooling layers reduce the spatial dimensions of the feature maps, lowering computational complexity and focusing on the most prominent features. MaxPooling is particularly effective in preserving the most critical information.

```python
layers.MaxPooling2D((2, 2)),
```


### Batch Normalization:
- **Reasoning**: Batch normalization layers are added after convolutional and fully connected layers to normalize the activations. This helps in stabilizing and accelerating the training process by reducing internal covariate shifts.

```python
layers.BatchNormalization(),
```

### Dropout:
- **Reasoning**: Dropout layers are used as a regularization technique to prevent overfitting. By randomly setting a fraction of input units to zero during training, the network becomes more robust and less likely to overfit the training data.

```python
layers.Dropout(0.3),
```

### Dense Layers:
- **Reasoning**: Dense layers towards the end of the network serve to integrate the features learned by the convolutional layers and make the final classification. The inclusion of regularizers and the swish activation function aids in preventing overfitting and improving performance.

```python
layers.Flatten(),
layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(0.001), kernel_initializer=HeNormal()),
layers.Dropout(0.5),
layers.BatchNormalization(),
layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(0.001), kernel_initializer=HeNormal()),
layers.Dropout(0.5),
layers.BatchNormalization(),
```

### Softmax Activation in Output Layer:
- **Reasoning**: The softmax activation function in the output layer converts the logits into probabilities for each class, enabling multi-class classification. Softmax activation function for \( K \) classes:
\[ \text{softmax}(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}} \], where:
- \( z = (z_1, z_2, ..., z_K) \) is the vector of logits (raw outputs) for each class.
- \( \text{softmax}(z)_j \) is the probability of class \( j \).
- The denominator is the sum of exponentiated logits across all classes, ensuring that the probabilities sum to 1.
  
```python
layers.Dense(8, activation='softmax')
```	    

So here is the final Architecture for the CNN model:


![image](https://github.com/Khaledayman9/Car-Logo-Classification/assets/105018459/2b0e07a1-d50c-401a-bd58-b95b6ea99d2d)


## Parameters:

### Learning Rate (Adam Optimizer):
- **Reasoning**: The Adam optimizer with a learning rate of 0.001 is chosen for its adaptive learning rate capabilities and efficient handling of sparse gradients. This optimizer combines the advantages of RMSprop and AdaGrad, making it suitable for training deep neural networks.

```python
model.compile(optimizer= Adam(learning_rate=0.001),
loss='categorical_crossentropy',
metrics=['accuracy'])
```

### Loss Function (Categorical Crossentropy):
- **Reasoning**: Categorical crossentropy is used as the loss function because it is appropriate for multi-class classification tasks. It measures the dissimilarity between the true labels and the predicted probabilities, providing a clear objective for the optimizer to minimize.


### ●	Batch Size and Epochs:
- **Reasoning**: A batch size of 64 is selected to provide a good balance between training speed and the stability of the gradient updates. Training for 200 epochs allows the model to learn effectively from the data, with enough iterations to converge to an optimal solution.
  
```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=200
)
```	 


# 9. Results:
The final model achieved a test accuracy of approximately 79.5%. However, the detailed classification report and confusion matrix indicated that the model's performance varied across different classes, with some brands being better recognized than others.
