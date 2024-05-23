# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# %%
train_set= ('./data_pre/mel_spectrogram_imgs/train')
test_set=('./data_pre/mel_spectrogram_imgs/test')
val_set= (r'./data_pre/mel_spectrogram_imgs/val')
 
# %%
train_datagen = image.ImageDataGenerator(rescale= 1./255)
val_datagen= image.ImageDataGenerator(rescale= 1./255)
test_datagen= image.ImageDataGenerator(rescale= 1./255)


# %%

train_generator = train_datagen.flow_from_directory(train_set,batch_size =128 ,class_mode = 'categorical')
test_generator = test_datagen.flow_from_directory(test_set,shuffle=True,batch_size =128 ,class_mode = 'categorical')
validation_generator = test_datagen.flow_from_directory(val_set,shuffle=True,batch_size =128 ,class_mode = 'categorical')

# %%
x_train, y_train = next(train_generator)
x_val,y_val= next(validation_generator)
x_test, y_test = next(test_generator)

# %%
import optuna
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import swish, relu, selu

def objective(trial):
    # Define the search space for hyperparameters
    num_conv_layers = trial.suggest_int('num_conv_layers', 3, 6)
    kernel_size = trial.suggest_categorical('kernel_size', [3,5,7])
    num_kernels = trial.suggest_int('num_kernels', 16, 128)
    stride = trial.suggest_int('stride', 1, 3)
    activation = trial.suggest_categorical('activation', ['relu', 'swish', 'selu'])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
    dense_dropout_rate = trial.suggest_uniform('dense_dropout_rate', 0.1, 0.5)
    dense_use_batch_norm = trial.suggest_categorical('dense_use_batch_norm', [True, False])
    num_neurons_per_layer = trial.suggest_int('num_neurons_per_layer', 10, 500)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    # Create the CNN model with hyperparameters
    model = Sequential()

    # Add the convolutional layers to the model
    for _ in range(num_conv_layers):
        model.add(Conv2D(num_kernels, kernel_size=kernel_size, padding='same', strides=stride))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())

    # Add the dense layers to the model
    for _ in range(num_dense_layers):
        model.add(Dense(num_neurons_per_layer))
        if dense_use_batch_norm:
            model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(dense_dropout_rate))

    model.add(Dense(10, activation='softmax'))

    # Compile the model with hyperparameters
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with hyperparameters
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.001, restore_best_weights=True)
    pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_loss')
    lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_lr=learning_rate/1000)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=150, callbacks=[early_stopping, pruning_callback,lr_on_plateau], verbose=1)

    # Return the validation accuracy as the objective value
    return history.history['val_acc'][-1]

# Create an Optuna study and optimize the objective function
sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=100)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=1000)


save_path = './opt_study/optuna_base_scratch_r0_TPE100_study.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(study, f)

# Print the best hyperparameters and objective value
best_params = study.best_params
best_value = study.best_value
print('Best Hyperparameters:', best_params)
print('Best Objective Value:', best_value)



