from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Define data paths
data_dir = 'C:/Users/inafe/OneDrive/Desktop/Plants - Copy'
checkpoint_dir = 'checkpoints'  # Directory where checkpoints are saved

# Verify GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generators - must be the same as in training
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Calculate steps per epoch
steps_per_epoch = int(np.ceil(train_generator.samples / train_generator.batch_size))
validation_steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))

# Create callbacks
checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5')
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Load the latest checkpoint
checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.h5'))
if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"Loading checkpoint: {latest_checkpoint}")
    model = tf.keras.models.load_model(latest_checkpoint)
    
    # Extract epoch number from checkpoint filename
    # Example filename: model_epoch_05_val_acc_0.8765.h5
    try:
        initial_epoch = int(os.path.basename(latest_checkpoint).split('epoch_')[1].split('_')[0])
        print(f"Resuming from epoch {initial_epoch}")
    except:
        # If filename parsing fails, start from epoch 0
        print("Couldn't determine the epoch number, starting from 0")
        initial_epoch = 0
    
    # Set the total number of epochs (initial + additional)
    total_epochs = initial_epoch + 10
    
    # Continue training
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        callbacks=[checkpoint_callback, early_stopping]
    )
    
    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model_resumed.h5')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
else:
    print("No checkpoints found. Please run the initial training first.")










""""
    \begin{figure}[ht]
    \centering
    % -- FIRST ROW --
    \includegraphics[width=0.3\textwidth]{Apple_p.jpg}
    \includegraphics[width=0.3\textwidth]{Apple_p2.jpg}
    \includegraphics[width=0.3\textwidth]{Banana_p.jpg}

    \vspace{2mm}

    % -- SECOND ROW --
    \includegraphics[width=0.3\textwidth]{Banana_p.jpg}
    \includegraphics[width=0.3\textwidth]{Strawberry_p.jpg}
    \includegraphics[width=0.3\textwidth]{Strawberry_p2.jpg}

    \vspace{2mm}

    % -- THIRD ROW --
    \includegraphics[width=0.3\textwidth]{Mango_p.jpg}
    \includegraphics[width=0.3\textwidth]{Mango_p2.jpg}
    \includegraphics[width=0.3\textwidth]{Grapes_p.jpg}

    \caption{Nine unseen fruit images obtained from the Internet. 
    Top row (left to right): \emph{Cherry}, \emph{Apple}, \emph{Banana}.
    Middle row: \emph{Banana}, \emph{Strawberries}, \emph{Strawberry}.
    Bottom row: \emph{Mangoes}, \emph{(Potentially Kiwifruit)}, \emph{Grapes}.
    These serve as the raw inputs for our model evaluation.}
    \label{fig:unseen_images}
\end{figure}
""""