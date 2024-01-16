
import torch

import argparse, os 
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split


import sagemaker
import argparse
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd


if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type = int, default =1)
    parser.add_argument("--learning-rate", type = float, default=0)
    parser.add_argument("--batch-size", type = int, default =32)

    parser.add_argument("--gpu-count", type=int, default=os.environ.get("SM_NUM_GPUs", 0))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_DATA", "./data"))
    parser.add_argument("--training", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])

    args,_ = parser.parse_known_args()

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    validation_dir = args.validation 

    # Create a DataFrame for the new directory structure
    df = pd.DataFrame(columns=['filepaths', 'class_id', 'dataset'])
    data_dir = "data"

    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            image_files = os.listdir(category_path)
            file_paths = [os.path.join(category_path, file) for file in image_files]
            class_id = category  # Assign a unique class ID to each category
            dataset = 'all'  # You can adjust this based on your dataset split

            df = pd.concat([df, pd.DataFrame({'filepaths': file_paths, 'class_id': class_id, 'dataset': dataset})])
            
    
    
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Update the 'dataset' column based on the split
    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'

    # Image Data Generators
    datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images)

    # Convert 'class_id' column to string type
    df['class_id'] = df['class_id'].astype(str)

    # Create generators for training and testing
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepaths',
        y_col='class_id',
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepaths',
        y_col='class_id',
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,  # No shuffling for the test set
        class_mode='categorical'
    )

        #LeNet Network Architecture
    
    model = Sequential()
    
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape= (224, 224,3)))
    
    model.add(AveragePooling2D())
    
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))   
    
    model.add(AveragePooling2D())
    
    model.add(Flatten())
    
    model.add(Dense(units=512, activation='relu'))
    
    model.add(Dense(units=512, activation='relu'))
    
    model.add(Dense(units=512, activation='relu'))
    
    model.add(Dense(units=4, activation = 'softmax'))
    
    print(model.summary())

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
    # Configure a MirroredStrategy for multi-GPU training
        strategy = tf.distribute.MirroredStrategy()
        print('Number of GPUs: {}'.format(len(gpus)))
    else:
        strategy = tf.distribute.get_strategy()
        print('No GPUs available, using CPU')
    
    # Compile and train the model
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        verbose=2
    )

    score = model.evaluate(test_generator, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])

    tf.saved_model.save(
        model,
        os.path.join(model_dir, 'model/1'),
        signatures={'inputs': model.input},
        options=None
    )



