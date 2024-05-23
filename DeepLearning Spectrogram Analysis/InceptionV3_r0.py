# %%
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import swish, relu,selu

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

train_set= ('./Experiments/data_pre/spec_imgs/train')
test_set=('./Experiments/data_pre/spec_imgs/test')
val_set= (r'./Experiments/data_pre/spec_imgs/val')

train_datagen = image.ImageDataGenerator(rescale= 1./255)
val_datagen= image.ImageDataGenerator(rescale= 1./255)
test_datagen= image.ImageDataGenerator(rescale= 1./255)


train_generator = train_datagen.flow_from_directory(train_set,batch_size =512 ,class_mode = 'categorical')
test_generator = test_datagen.flow_from_directory(test_set,shuffle=True,batch_size =128 ,class_mode = 'categorical')
validation_generator = test_datagen.flow_from_directory(val_set,shuffle=True,batch_size =128 ,class_mode = 'categorical')
x_train, y_train = next(train_generator)
x_val,y_val= next(validation_generator)
x_test, y_test = next(test_generator)

# %%
best_params = {'num_layers': 31, 'num_neurons': 6845, 'learning_rate': 0.00057624872164786}


# %%
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint


# Access the individual hyperparameters
learning_rate = best_params['learning_rate']
num_layers = best_params['num_layers']
num_neurons = best_params['num_neurons']

# Use the best hyperparameters in your code
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Freeze the layers in the base model
for layer in base_model.layers[:-num_layers]:
    layer.trainable = False

# Create a new model
model = Sequential()

# Add the VGG16 base model to the new model
model.add(base_model)
model.add(Flatten())
model.add(Dense(num_neurons, input_shape=(x_train.shape[1],)))
model.add(Activation(relu))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

# Compile the model with hyperparameters
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with hyperparameters
csv_logger = CSVLogger('./logs/InceptionV3_r0_training.csv')
# Define the checkpoint callback
checkpoint = ModelCheckpoint(filepath='./model_weights/InceptionV3_r0_model_weights.h5', 
                             monitor='val_loss', 
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min',
                             verbose=1)


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.0001, restore_best_weights=True)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=150, callbacks=[early_stopping,checkpoint,csv_logger], verbose=1)


# %%
import matplotlib.pyplot as plt

# Get the training and validation loss from the history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Get the training and validation accuracy from the history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('./img/InceptionV3_r0_loss_plot.png')

# Plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('./img/InceptionV3_r0_accuracy_plot.png')



# %%
model.evaluate(x_test, y_test)



# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Assuming you have the predicted labels and true labels
y_pred = model.predict(x_test).argmax(axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Calculate the F1 score
f1 = f1_score(y_true, y_pred, average='weighted')

# Plot the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print the F1 score
print('F1 Score:', f1)
plt.savefig('./img/InceptionV3_r0_confusion_matrix.png')


