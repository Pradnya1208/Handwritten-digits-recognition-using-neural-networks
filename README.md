<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Handwritten Digits recognition using Neural Networks</div>
<div align="center"><img src="https://github.com/Pradnya1208/Handwritten-digits-recognition-using-neural-networks/blob/main/output/intro.gif?raw=true" width="60%"></div>



## Overview:
Natural language processing (NLP) is a field of computer science, artificial intelligence concerned with the interactions between computers and human (natural) languages, and, in particular, concerned with programming computers to fruitfully process large natural language data.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.
<br>
We have done this project in 2 parts, first we have prerocessed the datset and in the second part we are doing the classification.

## Dataset:
[MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.
## Implementation:

**Libraries:**  `NumPy`  `pandas` `sklearn`  `Matplotlib` `tensorflow` `keras` `plotly`



### Data Exploration:
#### Displaying some images:
<img src="https://github.com/Pradnya1208/Handwritten-digits-recognition-using-neural-networks/blob/main/output/eda1.PNG?raw=true" width="50%">

#### Pre-processing:
```
X_train = X_train.reshape(60000, 28,28,1)
X_test = X_test.reshape(10000, 28,28,1)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train/=255.0
X_test/=255.0

#1D to 10 bins representing 10 digits
y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test,num_classes)
print(y_train.shape, y_test.shape)
```
```
(60000, 10) (10000, 10)
```
#### Building Neural Networks:
<img src="https://github.com/Pradnya1208/Handwritten-digits-recognition-using-neural-networks/blob/main/output/NN.PNG?raw=true" width="70%">

```
cnn.add(Conv2D(32 , kernel_size = (5,5), input_shape = (28,28,1),padding = "same", activation ="relu"))
# Maxpooling layer
cnn.add(MaxPooling2D())

cnn.add(Conv2D(64 , kernel_size = (5,5), padding = "same", activation ="relu"))
# Maxpooling layer
cnn.add(MaxPooling2D())

# Flatten the network
cnn.add(Flatten()) # because we have a fully connected network next


cnn.add(Dense(1024,activation = 'relu'))

# Output layer
cnn.add(Dense(10,activation = 'softmax'))
cnn.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
```
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        832       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 3136)              0         
_________________________________________________________________
dense (Dense)                (None, 1024)              3212288   
_________________________________________________________________
dense_1 (Dense)              (None, 10)                10250     
=================================================================
Total params: 3,274,634
Trainable params: 3,274,634
Non-trainable params: 0
_________________________________________________________________
```
<img src="https://github.com/Pradnya1208/Handwritten-digits-recognition-using-neural-networks/blob/main/output/acc-valacc.PNG?raw=true" width="40%">

Blue line indicates train accuracy and orange is for Validation Accuracy.
<br>

### Model evaluation:
```
pred  = np.argmax(pred, axis = 1)
```
<img src="https://github.com/Pradnya1208/Handwritten-digits-recognition-using-neural-networks/blob/main/output/result.PNG?raw=true" width="100%">

### Learnings:
`Convolutional Neural Networks` 






## References:
[Handwritten digits recognition](https://github.com/Pradnya1208/Handwritten-digits-recognition-using-neural-networks/blob/main/MNIST%20Handwritten%20Digits%20Recognition.ipynb)<br>

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner



[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]
