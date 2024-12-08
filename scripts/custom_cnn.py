from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, LeakyReLU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import json

class CustomCNN:
    """
    Class to handle CNN model creation, training, and evaluation with hyperparameter tuning.
    
    Attributes:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): Number of output classes for classification.
    """
    
    def __init__(self, input_shape=(128, 128, 3), num_classes=2):
        """
        Initializes the CNNHyperparameterTuning class with input shape and number of classes.

        Parameters:
            input_shape (tuple): Shape of the input images (height, width, channels).
            num_classes (int): Number of output classes for classification.
        """
        if len(input_shape) != 3:
            raise ValueError("Input shape must be a tuple of length 3: (height, width, channels).")
        if num_classes < 2:
            raise ValueError("Number of classes must be at least 2.")
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self, dropout_rate=0.5, l2_reg=0.001, batch_norm=False):
        """
        Builds a CNN model with specified hyperparameters.
    
        Parameters:
            dropout_rate (float): Rate of dropout for regularization.
            l2_reg (float): L2 regularization strength for Conv2D and Dense layers.
            batch_norm (bool): Whether to use batch normalization after convolution layers.
    
        Returns:
            tf.keras.Model: Compiled CNN model.
        """
        input_layer = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), kernel_regularizer=l2(l2_reg))(input_layer)
        if batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropout_rate)(x)
    
        x = Conv2D(64, (3, 3), kernel_regularizer=l2(l2_reg))(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropout_rate)(x)
    
        x = Conv2D(128, (3, 3), kernel_regularizer=l2(l2_reg))(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropout_rate)(x)
    
        x = Flatten()(x)
        x = Dense(128, kernel_regularizer=l2(l2_reg))(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dropout(dropout_rate)(x)
    
        # Corrected output layer based on the number of classes
        if self.num_classes == 2:
            output_layer = Dense(1, activation='sigmoid')(x)  # Single neuron for binary classification
        else:
            output_layer = Dense(self.num_classes, activation='softmax')(x)  # Softmax for multi-class
    
        model = Model(inputs=input_layer, outputs=output_layer)
        return model


    def train_model(self, train_dataset, val_dataset, epochs=50, 
                    dropout_rate=0.5, l2_reg=0.001, batch_norm=True, 
                    learning_rate=0.001):
        """
        Trains the CNN model with specified datasets and hyperparameters.

        Parameters:
            train_dataset (tf.data.Dataset): The training dataset.
            val_dataset (tf.data.Dataset): The validation dataset.
            epochs (int): Number of epochs to train the model.
            dropout_rate (float): Rate of dropout for regularization.
            l2_reg (float): L2 regularization strength for Conv2D and Dense layers.
            batch_norm (bool): Whether to use batch normalization after convolution layers.
            learning_rate (float): Learning rate for the Adam optimizer.

        Returns:
            history: Training history object containing loss and accuracy.
        """
        # Build and compile the model
        self.model = self.build_model(dropout_rate, l2_reg, batch_norm)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss='binary_crossentropy' if self.num_classes == 2 else 'categorical_crossentropy',
                           metrics=['accuracy'])

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        progress_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: 
                                           print(f"Epoch {epoch+1}: Loss={logs['loss']:.4f}, Accuracy={logs['accuracy']:.4f}"))

        # Train the model
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[early_stopping, lr_reduction, progress_callback]
        )
        return history

    def plot_training_history(self, history):
        """
        Plots the training and validation accuracy and loss.

        Parameters:
            history: Training history object containing loss and accuracy data.
        """
        # Extract accuracy and loss values
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Plot Accuracy
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy', color='blue')
        plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy', color='orange')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', color='blue')
        plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', color='orange')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def evaluate(self, test_dataset):
        """
        Evaluates the trained model on the test dataset.

        Parameters:
            test_dataset (tf.data.Dataset): The test dataset.

        Returns:
            tuple: Test loss and accuracy.
        """
        test_loss, test_accuracy = self.model.evaluate(test_dataset)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy

    def save_model(self, file_path='cnn_model.keras', save_format='keras'):
        """
        Saves the trained model.

        Parameters:
            file_path (str): Path to save the model.
            save_format (str): Format to save the model ('keras', 'h5', or 'tf').
        """
        if self.model:
            if save_format not in ['keras', 'h5', 'tf']:
                raise ValueError("Unsupported save format. Use 'keras', 'h5', or 'tf'.")
            self.model.save(file_path, save_format=save_format)
            print(f"Model saved to {file_path} in {save_format} format.")
        else:
            print("No model is available to save. Train the model first.")

    def log_hyperparameters(self, file_path='hyperparameter_log.json', **params):
        """
        Logs hyperparameters and training results to a JSON file.

        Parameters:
            file_path (str): Path to save the JSON file.
            params: Key-value pairs of hyperparameters and results.
        """
        with open(file_path, 'w') as file:
            json.dump(params, file)
        print(f"Hyperparameters logged to {file_path}")
