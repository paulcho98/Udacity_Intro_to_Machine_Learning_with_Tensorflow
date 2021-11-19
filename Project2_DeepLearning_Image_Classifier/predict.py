import argparse
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Neural Network Settings")
    
    parser.add_argument('image_path',
                        type=str,
                        help= 'Point to image file for prediction.')
    parser.add_argument('model_path',
                        type=str,
                        help= 'Point to model file as str.')
    parser.add_argument('--top_k',
                        type = int,
                        help = 'Number of predictions.')
    parser.add_argument('--category_names',
                        type = str,
                        help= 'Point to label.map json file')
    args = parser.parse_args()
    image_path = args.image_path
    model_path = args.model_path
    model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer}, compile = False)

    top_k = args.top_k
    if top_k == None:
        top_k = 1
    #label_map = args.category_names
    
    image_size = 224
    
    def process_image(image):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (image_size, image_size))
        image /= 255
        return image
    
    def predict(image_path, model, top_k):
        im = Image.open(image_path)
        image = np.asarray(im)
        image = np.expand_dims(process_image(image), axis = 0)
        predictions = model.predict(image)
        top_k_probs, top_k_classes = tf.math.top_k(predictions, k=top_k)
        top_k_classes = top_k_classes[0]+1
        return top_k_probs.numpy()[0], top_k_classes.numpy().astype(str)
    
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        probs, classes = predict(image_path, model, top_k)
        class_labels = [class_names[x] for x in classes]
        print("Class: ", class_labels)
        print("Probability: ", probs)    
    else:
        probs, classes = predict(image_path, model, top_k)
        print("Class: ", classes)
        print("Probability: ", probs)
    

    