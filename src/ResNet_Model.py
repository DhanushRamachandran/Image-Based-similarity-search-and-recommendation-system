import os
import sys
from keras.applications.resnet import ResNet50,preprocess_input
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.models import Sequential

model = ResNet50(input_shape=(224,224,3),weights="imagenet",include_top=False)
model.trainable = False
model = Sequential([model,GlobalMaxPooling2D()])
print(model.summary())
model.save(r"C:\Users\sudha\Desktop\dhanush\Personal DS\Computer Vision\Fashion recommendation system/ImgFeature_model.h5")

