# import all library
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
%matplotlib inline


# load fashion dataset using keras  ########
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()
print('Training dataset shape : ', train_X.shape, train_Y.shape)
print('Testing dataset shape : ', test_X.shape, test_Y.shape)

# find unique classes in the dataset
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of output classes : ', nClasses)
print('Output classes : ', classes)


plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Actual image : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Actual image : {}".format(test_Y[0]))


# reshape training data into 28*28
train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)
train_X.shape, test_X.shape


# type conversion float32 format
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
# convert pixel value in the rang of 0 to 1
train_X = train_X / 255
test_X = test_X / 255

# make category of the train and test data level
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])


# split data into train and validation of 80% and 20%
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape

# define batch size, # of epochs and number of classes for output
batch_size = 64
epochs = 20
num_classes = 10


# CNN Model Structure with 3 layers

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=(28,28,1)))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))         
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='relu'))     
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))


# CNN Model Summary
fashion_model.summary()

# compile cnn model
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# fit the model with training dataset
fashion_train_dropout = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))


# save the tramed model
fashion_model.save("fashion_trained_model.h5py")

# Evaluate test dataset on the model
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)


# test data loss and test data accuracy
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])



# Plot graph b/w accuracy and loss on training and validation dataset

accuracy = fashion_train_dropout.history['acc']
val_accuracy = fashion_train_dropout.history['val_acc']
loss = fashion_train_dropout.history['loss']
val_loss = fashion_train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, validity_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, validity_loss, 'b', label='Validation loss')
plt.title('Training and validation losses')
plt.legend()
plt.show()




# predict test classes with probabilities
predicted_classes = fashion_model.predict(test_X)


# find target classes
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, test_Y.shape


# plot some currect test images with prediction
correct = np.where(predicted_classes==test_Y)[0]
print ("Found %d corrected labels in test data" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted levels in test data {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()
	
	
# show incurrect prediction images
incorrect = np.where(predicted_classes!=test_Y)[0]
print ("Found %d incorrect labels in test data" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted levels in test data {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()

# show the precision, recall and f1-score for all classes
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))

