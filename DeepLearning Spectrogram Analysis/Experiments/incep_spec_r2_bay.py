# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# %%
train_set= ('./data_pre/spec_imgs/train')
test_set=('./data_pre/spec_imgs/test')
val_set= (r'./data_pre/spec_imgs/val')

# %%
train_datagen = image.ImageDataGenerator(rescale= 1./255)
val_datagen= image.ImageDataGenerator(rescale= 1./255)
test_datagen= image.ImageDataGenerator(rescale= 1./255)
 

# %%

train_generator = train_datagen.flow_from_directory(train_set,batch_size =512 ,class_mode = 'categorical')
test_generator = test_datagen.flow_from_directory(test_set,shuffle=True,batch_size =128 ,class_mode = 'categorical')
validation_generator = test_datagen.flow_from_directory(val_set,shuffle=True,batch_size =128 ,class_mode = 'categorical')

# %%
x_train, y_train = next(train_generator)
x_val,y_val= next(validation_generator)
x_test, y_test = next(test_generator)

# %%
import pickle
import optuna
from keras.applications import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import swish, relu,selu


def objective(trial):
    # Define the search space for hyperparameters
    layer_size_1 = trial.suggest_int('layer_size_1', 10, 4098) ######## modify the range
    layer_size_2 = trial.suggest_int('layer_size_2', 10, 4098) ######## modify the range
    layer_size_3 = trial.suggest_int('layer_size_3', 10, 4098) ######## modify the range

    dropout_rate_1 = trial.suggest_uniform('dropout_rate_1', 0.1, 0.5)
    dropout_rate_2 = trial.suggest_uniform('dropout_rate_2', 0.1, 0.5)
    dropout_rate_3 = trial.suggest_uniform('dropout_rate_3', 0.1, 0.5)
    activation = trial.suggest_categorical('activation', ['relu','swish','selu'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    # Create the MLP model with hyperparameters
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Freeze the layers in the base model
    for layer in base_model.layers[:-31]: #####   modify the number of layers
        layer.trainable = False

# Create a new model
    model = Sequential()

# Add the VGG16 base model to the new model
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(layer_size_1, input_shape=(x_train.shape[1],)))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate_1))
    

    model.add(Dense(layer_size_2))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate_2))


    model.add(Dense(layer_size_3))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate_3))

    model.add(Dense(10, activation='softmax'))


    # Compile the model with hyperparameters
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model with hyperparameters
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.001, restore_best_weights=True)
    pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_loss')
    lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_lr=learning_rate/1000)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=150, callbacks=[early_stopping, pruning_callback,lr_on_plateau], verbose=1)

    # Return the validation accuracy as the objective value
    return history.history['val_accuracy'][-1]

# Create an Optuna study and optimize the objective function

sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=50)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=150)


save_path = './opt_study/optuna_incep_spec_r2_bay_study.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(study, f)

# Print the best hyperparameters and objective value
best_params = study.best_params
best_value = study.best_value
print('Best Hyperparameters:', best_params)
print('Best Objective Value:', best_value)



