import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt', 'Pants', 'Sweatshirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Don't need this anymore
#plt.figure()
#plt.imshow(train_images[1])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#Normalizing data
train_images = train_images / 255.0
test_images = test_images / 255.0

#plt.figure(figsize=(10,10))
#for i in range(25, 50):
#	plt.subplot(5,5,i+1-25)
#	plt.xticks([])
#	plt.yticks([])
#	plt.grid(False)
#	plt.imshow(train_images[i], cmap=plt.cm.binary)
#	plt.xlabel(class_names[train_labels[i]])
#plt.show()

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)), #Flatten just takes the 28 by 28 dim array and turns it into a 28*28=784 dim array
	keras.layers.Dense(128, activation=tf.nn.relu), #Not sure why they chose relu activation type or used 128 nodes
	keras.layers.Dense(10, activation=tf.nn.softmax) #10 nodes because 10 possible outcomes and activation type softmax choses the highest value of all nodes in layer
])
#Takes an optimizer, loss function, and metrics to measure
model.compile(optimizer=tf.train.AdamOptimizer(),
				loss='sparse_categorical_crossentropy', #Not sure what this means or how they got it
				metrics=['accuracy'])

#Time to train the model on the training data
model.fit(train_images, train_labels, epochs=5)

#Now time to test the model on the test data that we have
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy: ', test_acc)

def plot_image(i, predictions_array, true_label, img):
	predictions_array, true_label, img = predictions_array[i], true_label[i], img[i] #I don't understand this line at all
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	
	plt.imshow(img, cmap=plt.cm.binary)
	
	predicted_label = np.argmax(predictions_array)
	if predicted_label==true_label:
		color = 'blue'
	else:
		color = 'red'
	
	plt.xlabel("{} {:2.0f} {}".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label], color=color))
	
def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	
	thisplot = plt.pie(predictions_array, labels=class_names)
	predicted_label = np.argmax(predictions_array)
	
	#thisplot[predicted_label].set_color('red') Getting a tuple index error here, so just gonna skip the color part of it
	#thisplot[true_label].set_color('blue')

predictions = model.predict(test_images)
#Displaying one prediction
#i = 0
#plt.figure(figsize=(6,3))
#plt.subplot(1,2,1)
#plot_image(i, predictions, test_labels, test_images)
#plt.subplot(1,2,2)
#plot_value_array(i, predictions, test_labels)
#plt.show()

#Displaying several predictions
num_rows = 5
num_cols = 5
num_images = 5*5
plt.figure(figsize=(2*2*num_cols, num_rows))
for i in range(num_images):
	plt.subplot(num_rows, 2*num_cols, 2*i+1)
	plot_image(i, predictions, test_labels, test_images)
	plt.subplot(num_rows, 2*num_cols, 2*i+2)
	plot_value_array(i, predictions, test_labels)
plt.show() #The pie graphs make the format really ugly, but whatever, we can work on that