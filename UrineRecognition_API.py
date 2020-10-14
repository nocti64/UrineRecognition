import argparse
import os
import json
import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image 
from keras.models import load_model

# model performance
# loss: 0.0598 - accuracy: 0.9850 - val_loss: 0.2976 - val_accuracy: 0.8958

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, help='path of image (including extension)')
    return parser.parse_args()

def load_img(img_path):
    img = image.load_img(img_path).resize((299,299))
    img_arr = image.img_to_array(img)
    img_arr = np.array([img_arr])  # Convert single image to a batch.
    return img_arr

def decode_prediction(pred):
    p=list(pred).index(max(pred))
    return p

def load_json(path):
    with open(path) as json_file:
        model_structure_json = json.load(json_file)
        return model_structure_json

if __name__ == '__main__':
    
    args = parse_args()
    labels_mapping = {0:"茶色", 1:"淺黃/透明黃", 2:"琥珀/蜜糖尿色", 3:"淡尿", 4:"泡沫尿"}
    
    img_arr = load_img(os.path.join(os.getcwd(), args.img_path))
    model_structure = load_json("./model/model.json")
    
    model = tf.keras.models.model_from_json(model_structure)
    model.load_weights("./model/model_weights.h5")
    pred = model.predict(img_arr).squeeze()
    print(pred)
    pred = decode_prediction(pred)
    print("Diagmostoc result:{}".format(labels_mapping[pred]))
        

    

    