import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

print("---------- versions ----------")
print("python version: " + sys.version)
print("numpy version: " + np.__version__)
print("matplotlib version: " + mpl.__version__)
print("tensorflow version: " + tf.__version__)
print()


####################################
# Classifying MNIST dataset
####################################
# program variables
showImages = True
datasetName = 'mnist'   # mnist or fashion_mnist
nnType = 'standard'     # standard or cnn
checkpoint_dir = 'trained_partial'       # directory checkpoint weights of model are saved
entire_save_file = 'trained_entire.h5'   # file where entire model is saved


####################################
# Downloading the data and splitting
####################################
print("---------- downloading data using keras ----------")
if datasetName == 'mnist':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
elif datasetName == 'fashion_mnist':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


####################################
# Exploring the data
####################################
print("---------- exploring data ----------")
print("x_train.shape: " + str(x_train.shape))
print("y_train.shape: " + str(y_train.shape))
print("x_test.shape: " + str(x_test.shape))
print("y_test.shape: " + str(y_test.shape))


def printSingleImage(image, label):
    plt.imshow(image.reshape((28,28)), cmap=plt.get_cmap("gray"))
    plt.title("Label: " + str(class_names[label]))
    plt.show()

def printManyImages(images, labels):
    plt.figure(figsize=(10,10))
    for i in range(len(images)):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel("Label: " + str(class_names[labels[i]]))
    plt.show()

if showImages:
    printSingleImage(x_train[0], y_train[0])
    printManyImages(x_train[:20], y_train[:20])


####################################
# Preparing data
####################################
print("---------- preparing data ----------")
# Keras API needs arrays to be 4D
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
input_shape = (28, 28, 1)

# normalise by mapping all image data pixels of vals [0, 255] to vals [0, 1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255


####################################
# Building and compiling the model
####################################
print("---------- building and compiling the model ----------")

def create_model():
    if nnType == 'standard':
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),       # input layer: 2D array of 28x28 to 1D array of 784
            tf.keras.layers.Dense(128, activation=tf.nn.relu),      # hidden layer: 128 node dense layer
            tf.keras.layers.Dense(10,  activation=tf.nn.softmax)    # output layer: 10 node softmax represent probabilities (sum of 10 nodes is 1)
        ])

    elif nnType == 'cnn': 
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10,  activation=tf.nn.softmax)
        ])
    
    model.compile(optimizer='adam',                         # optimiser: algirithm to adjust model parameters given the loss funtion
                  loss='sparse_categorical_crossentropy',   # loss function: how far are we from what we want?
                  metrics=['accuracy'])                     # metrics: measurements of how well we are doing

    return model

model = create_model()
print(model.summary())


####################################
# Training
####################################
print("---------- training ----------")
# checkpoint callback
checkpoint_path = checkpoint_dir + "/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

# train the model on the training data
# epochs is the number of times to iterate over the training dataset
# batch size is the number of training elements to send through before updating model parameters
model.fit(x_train, y_train, verbose=1, shuffle=True, epochs=5, batch_size=32, callbacks=[cp_callback])

# save entire model
model.save(entire_save_file)


####################################
# Testing
####################################
print("---------- testing ----------")

# fresh model with no training
model = create_model()
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Accuracy on test dataset with untrained model:', test_accuracy)

# loaded weights trained model
model.load_weights(checkpoint_path)    # needs to have a model with the same archetecture as original 
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Accuracy on test dataset with weight loaded model:', test_accuracy)

# loaded entire trained model
model = tf.keras.models.load_model(entire_save_file)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Accuracy on test dataset with entire loaded model:', test_accuracy)


####################################
# Individual predictions
####################################
print("---------- predicting a single image ----------")
testImage = x_test[0]
testLabel = y_test[0]
testImage = np.expand_dims(testImage, axis=0)
prediction = model.predict(testImage)

print("predictions of our test image: " + str(prediction))
print("maximum value: " + str(class_names[np.argmax(prediction)]))
print("actual label: " + str(class_names[testLabel]))

if showImages:
    printSingleImage(testImage, testLabel)


