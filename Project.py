from tensorflow import keras
import glob
import PIL
import PIL.Image as Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
class_names = ["buildings","glacier","forest","mountain","sea", "street"]

def get_imgs(fnames, label=None):
    data = [np.asarray(Image.open(fname)) for fname in fnames]
    for idx, d in enumerate(data):
        if(d.shape[0] != d.shape[1]):
            data.pop(idx)
    if label == None:
        return np.array(data)
    labels = np.full((len(data),1),label)
    return np.array(data), labels

def load_dataset(datadir=''):
    buildings, l1 = get_imgs(glob.glob(datadir + "/buildings/*"),0)
    glacier, l2 = get_imgs(glob.glob(datadir + "/glacier/*"),1)
    forest, l3 = get_imgs(glob.glob(datadir + "/forest/*"),2)
    mountain, l4 = get_imgs(glob.glob(datadir + "/mountain/*"),3)
    sea, l5 = get_imgs(glob.glob(datadir + "/sea/*"),4)
    street, l6 = get_imgs(glob.glob(datadir + "/street/*"),5)
    labels = np.concatenate((l1,l2,l3,l4,l5,l6))
    data = np.concatenate((buildings,glacier,forest,mountain,sea,street))
    return data,labels

def predict(img):
    prediction = model.predict(img.reshape((1,150,150,3)))
    return class_names[np.argmax(prediction,axis=1)[0]]

# Load training and test data
data,labels = load_dataset("archive/seg_train/seg_train")
test, tlabel = load_dataset("archive/seg_test/seg_test")
print(data.shape,labels.shape)
print(test.shape,tlabel.shape)

# Model Initialization
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(150,150, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(6,activation='softmax'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
model.summary()

# Model Training
history = model.fit(data, labels, epochs=20, validation_data=(test,tlabel))

# Plotting model accuracy vs epoch
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Load prediction images, unlabeled
images = get_imgs(glob.glob("archive/seg_pred/seg_pred/*"))

# load saved model
model = keras.models.load_model("5Layers_64.keras")

# Example of prediction on an image from model
img = images[26]
print(predict(img))
Image.fromarray(img)
