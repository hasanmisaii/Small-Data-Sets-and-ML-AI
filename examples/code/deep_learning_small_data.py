"""
Deep Learning with Small Datasets
=================================

This module demonstrates deep learning techniques specifically designed for
small datasets, including transfer learning, data augmentation, and
regularization strategies.

Author: Small Data ML Research
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.applications import VGG16, ResNet50
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow version:", tf.__version__)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Some examples will be skipped.")

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
    print("PyTorch version:", torch.__version__)
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Some examples will be skipped.")

import warnings
warnings.filterwarnings('ignore')

class SmallDataDeepLearning:
    """
    A class demonstrating deep learning techniques for small datasets.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the SmallDataDeepLearning class.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(random_state)
        if PYTORCH_AVAILABLE:
            torch.manual_seed(random_state)
    
    def load_image_dataset(self):
        """
        Load a small image dataset (digits dataset as proxy for images).
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Train/test splits of features and targets
        """
        # Load digits dataset (8x8 images of digits 0-9)
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Reshape to image format
        X = X.reshape(-1, 8, 8, 1)
        
        # Normalize pixel values
        X = X.astype('float32') / 16.0  # Max pixel value is 16
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        print("Loaded digits dataset:")
        print(f"  - Training samples: {X_train.shape[0]}")
        print(f"  - Test samples: {X_test.shape[0]}")
        print(f"  - Image shape: {X_train.shape[1:]}")
        print(f"  - Number of classes: {len(np.unique(y))}")
        
        return X_train, X_test, y_train, y_test
    
    def create_small_subset(self, X_train, y_train, samples_per_class=20):
        """
        Create a small subset of the training data.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        samples_per_class : int
            Number of samples per class to keep
            
        Returns:
        --------
        X_small, y_small : arrays
            Small subset of training data
        """
        X_small = []
        y_small = []
        
        for class_label in np.unique(y_train):
            class_indices = np.where(y_train == class_label)[0]
            selected_indices = np.random.choice(
                class_indices, 
                size=min(samples_per_class, len(class_indices)), 
                replace=False
            )
            X_small.extend(X_train[selected_indices])
            y_small.extend(y_train[selected_indices])
        
        X_small = np.array(X_small)
        y_small = np.array(y_small)
        
        print(f"Created small subset:")
        print(f"  - Samples: {X_small.shape[0]}")
        print(f"  - Samples per class: ~{samples_per_class}")
        
        return X_small, y_small
    
    def demonstrate_data_augmentation_images(self, X_train, y_train):
        """
        Demonstrate image data augmentation techniques.
        
        Parameters:
        -----------
        X_train : array-like
            Training images
        y_train : array-like
            Training labels
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping image augmentation demo.")
            return None, None
        
        print("\n" + "="*50)
        print("IMAGE DATA AUGMENTATION")
        print("="*50)
        
        # Create ImageDataGenerator for augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,      # Rotate images by up to 20 degrees
            width_shift_range=0.1,  # Shift images horizontally by up to 10%
            height_shift_range=0.1, # Shift images vertically by up to 10%
            zoom_range=0.1,         # Zoom in/out by up to 10%
            fill_mode='nearest'     # Fill in missing pixels
        )
        
        # Fit the generator
        datagen.fit(X_train)
        
        # Visualize original and augmented images
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        # Original images
        for i in range(5):
            axes[0, i].imshow(X_train[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original {y_train[i]}')
            axes[0, i].axis('off')
        
        # Augmented images
        augmented_images = []
        augmented_labels = []
        
        for i, (aug_img, aug_label) in enumerate(datagen.flow(X_train[:5], y_train[:5], batch_size=1)):
            axes[1, i].imshow(aug_img[0].squeeze(), cmap='gray')
            axes[1, i].set_title(f'Augmented {aug_label[0]}')
            axes[1, i].axis('off')
            
            augmented_images.append(aug_img[0])
            augmented_labels.append(aug_label[0])
            
            if i >= 4:  # Show 5 examples
                break
        
        plt.suptitle('Original vs Augmented Images')
        plt.tight_layout()
        plt.show()
        
        # Generate augmented dataset
        print("Generating augmented dataset...")
        batch_size = 32
        steps_per_epoch = len(X_train) // batch_size
        
        # Create larger augmented dataset
        X_augmented = []
        y_augmented = []
        
        for i, (batch_x, batch_y) in enumerate(datagen.flow(X_train, y_train, batch_size=batch_size)):
            X_augmented.extend(batch_x)
            y_augmented.extend(batch_y)
            
            if i >= steps_per_epoch:  # Generate one epoch worth of data
                break
        
        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)
        
        print(f"Original dataset size: {len(X_train)}")
        print(f"Augmented dataset size: {len(X_augmented)}")
        
        return X_augmented, y_augmented
    
    def build_simple_cnn(self, input_shape, num_classes, regularization=True):
        """
        Build a simple CNN with regularization for small datasets.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input images
        num_classes : int
            Number of output classes
        regularization : bool
            Whether to include regularization techniques
            
        Returns:
        --------
        model : keras.Model
            Compiled CNN model
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot build CNN.")
            return None
        
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization() if regularization else layers.Lambda(lambda x: x),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25) if regularization else layers.Lambda(lambda x: x),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization() if regularization else layers.Lambda(lambda x: x),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25) if regularization else layers.Lambda(lambda x: x),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01) if regularization else None),
            layers.Dropout(0.5) if regularization else layers.Lambda(lambda x: x),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def compare_regularization_in_deep_learning(self, X_train, X_test, y_train, y_test):
        """
        Compare models with and without regularization.
        
        Parameters:
        -----------
        X_train, X_test, y_train, y_test : arrays
            Training and test data
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping regularization comparison.")
            return
        
        print("\n" + "="*50)
        print("REGULARIZATION IN DEEP LEARNING")
        print("="*50)
        
        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(y_train))
        
        # Build models with and without regularization
        model_regular = self.build_simple_cnn(input_shape, num_classes, regularization=True)
        model_no_regular = self.build_simple_cnn(input_shape, num_classes, regularization=False)
        
        print("Model with regularization:")
        model_regular.summary()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
        ]
        
        # Train both models
        epochs = 50
        
        print("\nTraining model WITH regularization...")
        history_regular = model_regular.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        print("Training model WITHOUT regularization...")
        history_no_regular = model_no_regular.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        # Plot training histories
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training accuracy
        axes[0, 0].plot(history_regular.history['accuracy'], label='With Regularization')
        axes[0, 0].plot(history_no_regular.history['accuracy'], label='Without Regularization')
        axes[0, 0].set_title('Training Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation accuracy
        axes[0, 1].plot(history_regular.history['val_accuracy'], label='With Regularization')
        axes[0, 1].plot(history_no_regular.history['val_accuracy'], label='Without Regularization')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training loss
        axes[1, 0].plot(history_regular.history['loss'], label='With Regularization')
        axes[1, 0].plot(history_no_regular.history['loss'], label='Without Regularization')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation loss
        axes[1, 1].plot(history_regular.history['val_loss'], label='With Regularization')
        axes[1, 1].plot(history_no_regular.history['val_loss'], label='Without Regularization')
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Regularization Comparison in Deep Learning')
        plt.tight_layout()
        plt.show()
        
        # Final evaluation
        regular_acc = model_regular.evaluate(X_test, y_test, verbose=0)[1]
        no_regular_acc = model_no_regular.evaluate(X_test, y_test, verbose=0)[1]
        
        print(f"\nFinal Test Accuracy:")
        print(f"  With Regularization: {regular_acc:.3f}")
        print(f"  Without Regularization: {no_regular_acc:.3f}")
        
        # Calculate overfitting metrics
        regular_train_acc = max(history_regular.history['accuracy'])
        regular_val_acc = max(history_regular.history['val_accuracy'])
        regular_gap = regular_train_acc - regular_val_acc
        
        no_regular_train_acc = max(history_no_regular.history['accuracy'])
        no_regular_val_acc = max(history_no_regular.history['val_accuracy'])
        no_regular_gap = no_regular_train_acc - no_regular_val_acc
        
        print(f"\nOverfitting Analysis (Train-Val Gap):")
        print(f"  With Regularization: {regular_gap:.3f}")
        print(f"  Without Regularization: {no_regular_gap:.3f}")
        
        return model_regular, model_no_regular
    
    def demonstrate_transfer_learning(self, X_train, X_test, y_train, y_test):
        """
        Demonstrate transfer learning for small datasets.
        
        Parameters:
        -----------
        X_train, X_test, y_train, y_test : arrays
            Training and test data
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping transfer learning demo.")
            return
        
        print("\n" + "="*50)
        print("TRANSFER LEARNING FOR SMALL DATASETS")
        print("="*50)
        
        # For demonstration, we'll upscale our 8x8 images to 32x32 to use with pre-trained models
        # In practice, you'd start with higher resolution images
        X_train_upscaled = tf.image.resize(X_train, [32, 32])
        X_test_upscaled = tf.image.resize(X_test, [32, 32])
        
        # Convert grayscale to RGB by repeating channels
        X_train_rgb = tf.repeat(X_train_upscaled, 3, axis=-1)
        X_test_rgb = tf.repeat(X_test_upscaled, 3, axis=-1)
        
        print("Upscaled images for transfer learning:")
        print(f"  Original shape: {X_train.shape}")
        print(f"  Upscaled shape: {X_train_rgb.shape}")
        
        # Load pre-trained VGG16 model
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(32, 32, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classifier on top
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(np.unique(y_train)), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Transfer learning model architecture:")
        model.summary()
        
        # Train the model
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        print("\nTraining transfer learning model...")
        history = model.fit(
            X_train_rgb, y_train,
            validation_data=(X_test_rgb, y_test),
            epochs=30,
            batch_size=16,  # Smaller batch size for small datasets
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model
        test_accuracy = model.evaluate(X_test_rgb, y_test, verbose=0)[1]
        print(f"\nTransfer Learning Test Accuracy: {test_accuracy:.3f}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Transfer Learning - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Transfer Learning - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return model
    
    def demonstrate_pytorch_small_data(self, X_train, X_test, y_train, y_test):
        """
        Demonstrate PyTorch techniques for small datasets.
        
        Parameters:
        -----------
        X_train, X_test, y_train, y_test : arrays
            Training and test data
        """
        if not PYTORCH_AVAILABLE:
            print("PyTorch not available. Skipping PyTorch demo.")
            return
        
        print("\n" + "="*50)
        print("PYTORCH TECHNIQUES FOR SMALL DATASETS")
        print("="*50)
        
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Define a simple CNN in PyTorch
        class SimpleCNN(nn.Module):
            def __init__(self, input_shape, num_classes):
                super(SimpleCNN, self).__init__()
                
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                
                self.pool = nn.MaxPool2d(2)
                self.dropout_conv = nn.Dropout2d(0.25)
                self.dropout_fc = nn.Dropout(0.5)
                
                # Calculate flattened size
                conv_output_size = self._get_conv_output_size(input_shape)
                
                self.fc1 = nn.Linear(conv_output_size, 128)
                self.fc2 = nn.Linear(128, num_classes)
                
            def _get_conv_output_size(self, shape):
                with torch.no_grad():
                    dummy_input = torch.zeros(1, *shape)
                    x = self.pool(torch.relu(self.bn1(self.conv1(dummy_input))))
                    x = self.pool(torch.relu(self.bn2(self.conv2(x))))
                    return int(np.prod(x.size()))
            
            def forward(self, x):
                x = self.pool(torch.relu(self.bn1(self.conv1(x))))
                x = self.dropout_conv(x)
                x = self.pool(torch.relu(self.bn2(self.conv2(x))))
                x = self.dropout_conv(x)
                
                x = x.view(x.size(0), -1)  # Flatten
                x = torch.relu(self.fc1(x))
                x = self.dropout_fc(x)
                x = self.fc2(x)
                
                return x
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = SimpleCNN(X_train.shape[1:], len(np.unique(y_train))).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        print("PyTorch CNN Architecture:")
        print(model)
        
        # Training loop
        num_epochs = 30
        train_losses = []
        train_accuracies = []
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            
            scheduler.step(epoch_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        test_accuracy = 100 * correct / total
        print(f'\nPyTorch CNN Test Accuracy: {test_accuracy:.2f}%')
        
        # Plot training progress
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('PyTorch Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies)
        plt.title('PyTorch Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return model


def main():
    """
    Main function to run deep learning examples for small datasets.
    """
    print("DEEP LEARNING WITH SMALL DATASETS")
    print("="*60)
    
    # Initialize the class
    dl_examples = SmallDataDeepLearning(random_state=42)
    
    # Load dataset
    X_train, X_test, y_train, y_test = dl_examples.load_image_dataset()
    
    # Create small subset for extreme small data scenarios
    X_train_small, y_train_small = dl_examples.create_small_subset(
        X_train, y_train, samples_per_class=30
    )
    
    print("\n🔍 Running Deep Learning Small Data Analysis...")
    
    # 1. Data Augmentation for Images
    if TENSORFLOW_AVAILABLE:
        X_augmented, y_augmented = dl_examples.demonstrate_data_augmentation_images(
            X_train_small, y_train_small
        )
    
    # 2. Regularization Comparison
    if TENSORFLOW_AVAILABLE:
        model_reg, model_no_reg = dl_examples.compare_regularization_in_deep_learning(
            X_train_small, X_test, y_train_small, y_test
        )
    
    # 3. Transfer Learning
    if TENSORFLOW_AVAILABLE:
        transfer_model = dl_examples.demonstrate_transfer_learning(
            X_train_small, X_test, y_train_small, y_test
        )
    
    # 4. PyTorch Implementation
    if PYTORCH_AVAILABLE:
        pytorch_model = dl_examples.demonstrate_pytorch_small_data(
            X_train_small, X_test, y_train_small, y_test
        )
    
    print("\n✅ Deep learning demonstrations completed!")
    print("\nKey Takeaways for Small Data Deep Learning:")
    print("1. Data augmentation is crucial for small image datasets")
    print("2. Strong regularization (dropout, batch norm, weight decay) prevents overfitting")
    print("3. Transfer learning leverages pre-trained knowledge")
    print("4. Early stopping prevents overtraining")
    print("5. Learning rate scheduling improves convergence")
    print("6. Smaller batch sizes often work better with limited data")


if __name__ == "__main__":
    main()