import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# configuration
DATASET_DIR = "data/train"  # folder containing your class subfolders
IMG_SIZE = 128              # resize all images to 128x128
BATCH_SIZE = 32
EPOCHS = 15

# load images and labels
images = []
labels = []

class_names = sorted(os.listdir(DATASET_DIR))  # get class names from folder names
print("loading images...")

for label, class_name in enumerate(class_names):
    class_dir = os.path.join(DATASET_DIR, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        try:
            img = cv2.imread(img_path)                 # read the image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # resize it
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"skipping {img_path}: {e}")

images = np.array(images) / 255.0  # normalize pixel values
labels = np.array(labels)
print(f"loaded {len(images)} images across {len(class_names)} classes\n")

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# create data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# build the cnn model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()  # show the model structure

# train the model
print("\nstarting training...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test)
)

# evaluate the model
print("\nevaluating on test data...")
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nclassification report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("confusion matrix - nutritional deficiencies")
plt.ylabel("true label")
plt.xlabel("predicted label")
plt.show()

# plot training accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title("accuracy over epochs")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()

# plot training loss
plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title("loss over epochs")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

# save the model
model.save("skin_deficiency_cnn_model.h5")  # save trained model
print("model successfully saved as skin_deficiency_cnn_model.h5") # WE DID IT!
