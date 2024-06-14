# Car-Logo-Classification
A CNN model that can accurately identify which car brand a logo belongs to among eight possible brands: Hyundai, Lexus, Mazda, Mercedes, Opel, Skoda, Toyota, and Volkswagen. This classification task is a subset of image recognition, a fundamental problem in computer vision.


# Problem:
The problem at hand is the classification of car brand logos using a Convolutional Neural Network (CNN). The task involves training a model to identify which car brand a logo belongs to among eight possible brands: Hyundai, Lexus, Mazda, Mercedes, Opel, Skoda, Toyota, and Volkswagen. This classification task is a subset of image recognition, a well-known problem in computer vision.

# Aim:
The primary aim of this task is to develop an accurate and robust model capable of classifying car brand logos from images. Achieving this goal can have practical applications in various fields, including automated quality control in manufacturing, visual marketing, and brand monitoring on social media.

# Importance:
1.	Automation: Automated logo classification can significantly reduce the time and effort required for manual identification and sorting of images.
   
2.	Scalability: A reliable model can handle large datasets, making it feasible to analyze extensive collections of images, which is impractical manually.
   
3.	Precision Marketing: Knowing where and how often brand logos appear can help companies tailor their marketing strategies.
   
4.	Intellectual Property Protection: Detecting unauthorized use of logos can help in protecting brand identity and intellectual property rights.

# Challenges:

1.	Variability in Logos: Logos might vary in color, size, orientation, and context in which they appear, making consistent classification challenging.
   
2.	Data Imbalance: Some brands may have more images available than others, leading to an imbalanced dataset that can bias the model.
   
3.	Overfitting: Given the limited number of classes and potential for many similar features among logos, the model might overfit to the training data.

4.	Generalization: Ensuring that the model performs well on unseen data (generalization) is crucial but challenging due to possible variations not represented in the training set.




# Methodology

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


# Dataset:
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




#. Results:
The final model achieved a test accuracy of approximately 79.5%. However, the detailed classification report and confusion matrix indicated that the model's performance varied across different classes, with some brands being better recognized than others.
