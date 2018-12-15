import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

model = load_model('fashion_mnist.h5')
i = 0
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
	
	plt.xlabel("{} {:2.0f}% {}".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label], color=color))
	
def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array[i], true_label[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	predicted_label = np.argmax(predictions_array)
	colors = ['gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray']
	colors[predicted_label] = 'green'
	thisplot = plt.pie(predictions_array, colors=colors)
	#thisplot[predicted_label].set_color('red') Getting a tuple index error here, so just gonna skip the color part of it
	#thisplot[true_label].set_color('blue')

predictions = model.predict(test_images[i])
#Displaying one prediction
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()