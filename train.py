import os
from PIL import Image
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

# Chemins des dossiers d'entraînement et de test
train_folder = 'data/MNIST - JPG - training'
test_folder = 'data/MNIST - JPG - testing'

# Listes pour stocker les images et les étiquettes d'entraînement
train_images = []
train_labels = []

# Parcourir les sous-dossiers correspondant aux chiffres de 0 à 9
for digit in range(10):
    digit_folder = os.path.join(train_folder, str(digit))
    
    # Parcourir les fichiers d'images dans chaque sous-dossier
    for filename in os.listdir(digit_folder):
        if filename.endswith(".jpg"):
            # Chemin complet de l'image
            img_path = os.path.join(digit_folder, filename)
            
            # Charger l'image, la convertir en niveaux de gris et la redimensionner
            img = Image.open(img_path).convert('L')
            img = img.resize((28, 28))
            
            # Convertir l'image en tableau numpy et normaliser les pixels
            img_array = np.array(img) / 255.0
            
            # Ajouter l'image et l'étiquette aux listes
            train_images.append(img_array)
            train_labels.append(digit)

# Convertir les listes en tableaux numpy
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Chargement des données de test
test_images = []
test_labels = []

# Parcourir les sous-dossiers correspondant aux chiffres de 0 à 9
for digit in range(10):
    digit_folder = os.path.join(test_folder, str(digit))
    
    # Parcourir les fichiers d'images dans chaque sous-dossier
    for filename in os.listdir(digit_folder):
        if filename.endswith(".jpg"):
            # Chemin complet de l'image
            img_path = os.path.join(digit_folder, filename)
            
            # Charger l'image, la convertir en niveaux de gris et la redimensionner
            img = Image.open(img_path).convert('L')
            img = img.resize((28, 28))
            
            # Convertir l'image en tableau numpy et normaliser les pixels
            img_array = np.array(img) / 255.0
            
            # Ajouter l'image et l'étiquette aux listes
            test_images.append(img_array)
            test_labels.append(digit)

# Convertir les listes en tableaux numpy
test_images = np.array(test_images)
test_labels = np.array(test_labels)


# Vérifier les dimensions des tableaux
print("Train Images Shape:", train_images.shape)
print("Train Labels Shape:", train_labels.shape)
print("Test Images Shape:", test_images.shape)

# Définition du modèle
def build_model():
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='random_uniform', input_shape=(28, 28, 1)))
    cnn.add(keras.layers.Dropout(0.5))
    cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

    cnn.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='random_uniform'))
    cnn.add(keras.layers.Dropout(0.5))
    cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

    cnn.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='random_uniform'))
    cnn.add(keras.layers.Dropout(0.5))
    cnn.add(keras.layers.MaxPool2D(pool_size=(3, 3)))

    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(units=128, activation='relu'))
    cnn.add(keras.layers.Dense(units=10, activation='softmax'))
    
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return cnn

# Construction du modèle
cnn = build_model()

# Définition d'une fonction pour afficher les courbes d'apprentissage
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Entraînement du modèle avec suivi des courbes d'apprentissage
history = cnn.fit(train_images, train_labels, epochs=50, validation_split=0.2, verbose=1)
plot_history(history)

# Prédiction sur les données de test
y_pred = np.argmax(cnn.predict(test_images), axis=-1)

# Sauvegarde du modèle
cnn.save('mnist_model.h5')

