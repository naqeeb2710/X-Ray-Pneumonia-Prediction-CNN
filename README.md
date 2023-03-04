# X-Ray-Pneumonia-Prediction-CNN
This project uses a convolutional neural network (CNN) to classify chest X-rays as either normal or showing signs of pneumonia. The model is trained on the [Chest X-Ray Images (Pneumonia) dataset from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The goal of this project is to develop a model that can accurately identify pneumonia in chest X-rays, which can help doctors and radiologists quickly and accurately diagnose patients.

# Getting Started
Clone the repository
``` 
git clone https://github.com/naqeeb2710/X-Ray-Pneumonia-Prediction-CNN.git
```  
Change the Directory
``` 
cd X-Ray-Pneumonia-Prediction-CNN
```
Install requirements
```
pip install -r requirements.txt
```
Run the python script to run the local flask server
```
python app.py
```
## Requirements
```
Python version 3.7
Tensorflow 2.x 
Keras 
Matplotlib 
Numpy 
Sklearn 
OpenCV 
```

# Data Preprocessing
The dataset contains two classes, Normal and Pneumonia. <br/>
The dataset is divided into two sets, one for training and the other for testing. <br/>
The images are resized and normalised before being fed into the model <br/>

# Model architecture
The model consists of a series of convolutional, max pooling, and dense layers. Dropout layers are used to prevent overfitting.

# Training and Evaluation
The model is trained for 150 epochs with a batch size of 131. <br/>
The training dataset is used to train the model, and the test dataset is used to evaluate the performance of the model. <br/>
The accuracy and loss are plotted to visualise the training process. <br/>

# Results
The model achieved an accuracy of 96% on the test dataset.

# Contributions
This project is open to contributions, feel free to fork the repository and make pull requests.




