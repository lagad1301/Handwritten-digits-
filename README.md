


pip install numpy 
 pip install matplotlib
 pip install tensorflow


***dwpendence***
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

learn = tf.contrib.learn 

tf.logging.set_verbosity(tf.logging.ERROR) 



***import dataset***
mnist = learn.datasets.load_dataset('mnist') 
data = mnist.train.images 
labels = np.asarray(mnist.train.labels, dtype=np.int32) 
test_data = mnist.test.images 
test_labels = np.asarray(mnist.test.labels, dtype=np.int32) 


***making dataset***
max_examples = 10000
data = data[:max_examples] 
labels = labels[:max_examples] 


***display dataset using matplotlib***
def display(i): 
	img = test_data[i] 
	plt.title('label : {}'.format(test_labels[i])) 
	plt.imshow(img.reshape((28, 28))) 
	
# image in TensorFlow is 28 by 28 px 
display(0) 

***fitting data using linear clissifier***
feature_columns = learn.infer_real_valued_columns_from_input(data) 
classifier = learn.LinearClassifier(n_classes=10, 
									feature_columns=feature_columns) 
classifier.fit(data, labels, batch_size=100, steps=1000)
 

***evaluate accuramcy***
classifier.evaluate(test_data, test_labels) 
print(classifier.evaluate(test_data, test_labels)["accuracy"]) 


***data prdicting***
prediction = classifier.predict(np.array([test_data[0]], 
										dtype=float), 
										as_iterable=False) 
print("prediction : {}, label : {}".format(prediction, 
	test_labels[0]) ) 

***full code ***
# importing libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

learn = tf.contrib.learn 
tf.logging.set_verbosity(tf.logging.ERROR)\ 

# importing dataset using MNIST 
# this is how mnist is used mnist contain test and train dataset 
mnist = learn.datasets.load_dataset('mnist') 
data = mnist.train.images 
labels = np.asarray(mnist.train.labels, dtype = np.int32) 
test_data = mnist.test.images 
test_labels = np.asarray(mnist.test.labels, dtype = np.int32) 

max_examples = 10000
data = data[:max_examples] 
labels = labels[:max_examples] 

# displaying dataset using Matplotlib 
def display(i): 
	img = test_data[i] 
	plt.title('label : {}'.format(test_labels[i])) 
	plt.imshow(img.reshape((28, 28))) 
	
# img in tf is 28 by 28 px 
# fitting linear classifier 
feature_columns = learn.infer_real_valued_columns_from_input(data) 
classifier = learn.LinearClassifier(n_classes = 10, 
									feature_columns = feature_columns) 
classifier.fit(data, labels, batch_size = 100, steps = 1000) 

# Evaluate accuracy 
classifier.evaluate(test_data, test_labels) 
print(classifier.evaluate(test_data, test_labels)["accuracy"]) 

prediction = classifier.predict(np.array([test_data[0]], 
										dtype=float), 
										as_iterable=False) 
print("prediction : {}, label : {}".format(prediction, 
	test_labels[0]) ) 

if prediction == test_labels[0]: 
	display(0) 


