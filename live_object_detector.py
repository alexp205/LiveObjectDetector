from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
import os
import re

from numpy import genfromtxt
from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

use_saved_model = False
saved_model_dir = ""

def loadImageData():
    """
    Loads both training and testing images as well as corresponding labels (in csv format) 
    into np arrays. The images are of tennis balls, lemons, and mangos with the 
    corresponding labels being:
      - tennis balls = 0
      - mangos = 1
      - lemons = 2
    """

    # DEBUG
    print("in loadImageData")
    
    # Load Training Images
    #tr_relpaths = getImagePaths(os.getcwd() + "/Data/Images/train")
    #tr_data = parseImages(tr_relpaths, 1305)
    # DEBUG
    tr_relpaths = getImagePaths(os.getcwd() + "/Data/Images/train_mini")
    tr_data = parseImages(tr_relpaths, 15)

    # DEBUG
    print("training features (images) loaded")
    
    # Load Training Labels
    #train_labels = genfromtxt("./Data/train_labels.csv", delimiter=',')
    # DEBUG
    train_labels = genfromtxt("./Data/train_mini_labels.csv", delimiter=',')
    tr_labels = train_labels.astype(int)

    # DEBUG
    print("training labels loaded")
    
    # Load Testing Images
    #tst_relpaths = getImagePaths(os.getcwd() + "/Data/Images/test")
    #tst_data = parseImages(tst_relpaths, 129)
    # DEBUG
    tst_relpaths = getImagePaths(os.getcwd() + "/Data/Images/test_mini")
    tst_data = parseImages(tst_relpaths, 6)

    # DEBUG
    print("testing features (images) loaded")
    
    # Load Testing Labels
    #test_labels = genfromtxt("./Data/test_labels.csv", delimiter=',')
    # DEBUG
    test_labels = genfromtxt("./Data/test_mini_labels.csv", delimiter=',')
    tst_labels = test_labels.astype(int)

    # DEBUG
    print("testing labels loaded")
    
    return tr_data, tr_labels, tst_data, tst_labels

def parseImages(filenames, size):
    """
    Reads in images and reshapes them to correct numpy array shape expected by cnn
    """
    
    data = np.zeros(shape=(size,400,400,3))
    with tf.Session() as sess:
        for idx, filename in enumerate(filenames):

            # DEBUG
            print("parsing image file: ", filename)

            img_file = tf.read_file(filename)
            img_decoded = tf.image.decode_image(img_file, channels=3)
            img_resized = tf.image.resize_image_with_crop_or_pad(img_decoded, 400, 400)
            img_resized.set_shape([400,400,3])
            img_raw = img_resized.eval(session=sess)
            data[idx] = img_raw

    return data

def getImagePaths(img_path):
    """
    Gets the relative paths for all image files in the specified directory
    """

    # DEBUG
    print("img_path: ", img_path)

    fileset = set()
    root = os.getcwd()
    
    for folder,_,files in os.walk(img_path):
        for filename in files:

            # DEBUG
            #print("file: ", filename)

            rel_dir = os.path.relpath(folder, root)
            rel_file = os.path.join(rel_dir, filename)
            fileset.add(rel_file)
    
    fileset = list(fileset)
    fileset = human_sort(fileset)
    
    return fileset

# Source: https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
def human_sort(list_to_sort):
    """
    Sorts a list according to human/natural sorting
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanumeric_key = lambda key: [ convert(part) for part in re.split('([0-9]+)', key) ]
    
    return sorted(list_to_sort, key = alphanumeric_key)
    
def cnnModelFxn(features, labels, mode):
    """
    Model function for the live object detector CNN. It uses two convolution modules 
    before consolidating values and outputting the classification configuration.
    """
      
    # Input Layer
    # -----------
    # In order to properly process images, they must first be reshaped and represented 
    # in the following format:
    #     [batch_size, width, height, channels]
    # Images are 400x400 pixels, and have three color channels
    input_layer = tf.reshape(features["x"], [-1, 400, 400, 3])

    # Convolutional Layer #1
    # ----------------------
    # Computes 32 features using a 50x50 filter with ReLU activation, padding is added to 
    # preserve width and height
    # Input Tensor: [batch_size, 400, 400, 3]
    # Output Tensor: [batch_size, 400, 400, 32]
    conv_1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=50,
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # ----------------
    # First max pooling layer with a 10x10 filter and stride of 10
    # Input Tensor: [batch_size, 400, 400, 32]
    # Output Tensor: [batch_size, 40, 40, 32]
    pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=10, strides=10)

    # Convolutional Layer #2
    # ----------------------
    # Computes 64 features using a 5x5 filter, padding is added to preserve width and height
    # Input Tensor: [batch_size, 40, 40, 32]
    # Output Tensor: [batch_size, 40, 40, 64]
    conv_2 = tf.layers.conv2d(
        inputs=pool_1,
        filters=64,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # ----------------
    # Second max pooling layer with a 5x5 filter and stride of 5
    # Input Tensor: [batch_size, 40, 40, 64]
    # Output Tensor: [batch_size, 8, 8, 64]
    pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=5, strides=5)

    # Flatten tensor into a batch of vectors
    # Input Tensor: [batch_size, 8, 8, 64]
    # Output Tensor: [batch_size, 8 * 8 * 64]
    pool_2_flat = tf.reshape(pool_2, [-1, 8 * 8 * 64])

    # Dense Layer
    # -----------
    # Densely connected layer with 1024 neurons
    # Input Tensor: [batch_size, 8 * 8 * 64]
    # Output Tensor: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool_2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation with 0.4 chance
    is_train = mode == tf.estimator.ModeKeys.TRAIN
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=is_train)

    # Logits Layer
    # ------------
    # Input Tensor: [batch_size, 1024]
    # Output Tensor: [batch_size, 3]
    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Training Configurations
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Evaluation Configurations
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def serving_input_receiver_fxn():
    holder = tf.placeholder(dtype=tf.float32,
                            shape=[None, 8],
                            name="input_placeholder_tensor")
    inputs = {"x": holder}
    
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def main(argv=None):
    if not use_saved_model:
        # DEBUG
        print("---------- PERFORMING TRAINING AND MODEL SAVING ----------")

        # Load Data
        train_data, train_labels, test_data, test_labels = loadImageData()

        # DEBUG
        print("PROGRESS - data loaded")

        # Create Estimator
        live_image_classifier = tf.estimator.Estimator(
            model_fn=cnnModelFxn, model_dir="live_object_detector_model")

        # DEBUG
        print("PROGRESS - estimator created")

        # Setup Logging
        # -------------
        # Log values in "softmax_tensor" with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

        # DEBUG
        print("PROGRESS - logging setup")
        print("PROGRESS - starting training")

        # Train
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=8,
            num_epochs=None,
            shuffle=True)
        live_image_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])

        # DEBUG
        print("PROGRESS - training complete")

        export_dir = live_image_classifier.export_savedmodel(export_dir_base="TEST_live_object_detector_model",
                                                             serving_input_receiver_fn=serving_input_receiver_fxn)

        print("NOTE: model saved at the following location, please mark down for future use")
        print(export_dir)

        # DEBUG
        print("PROGRESS - starting testing")

        # Evaluate and Display Results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_data},
            y=test_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = live_image_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

        # DEBUG
        print("PROGRESS - testing complete")
    else:
        # DEBUG
        print("---------- USING SAVED MODEL ----------")

        predict_fxn = predictor.from_saved_model(saved_model_dir)

    # DEBUG
    #print("PROGRESS - about to visualize")
      
"""
    # Visualize
    cap = cv2.VideoCapture(0)
    while True:
        ret, image_np = cap.read()
        image_np = cv2.resize(image_np, (400, 400))
        # Expand dimensions since the model expects images to have shape: [1, 400, 400, 3]
        predict_data = np.expand_dims(image_np, axis=0)
        # Prediction
        if not use_saved_model:
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": predict_data},
            num_epochs=1,
            shuffle=False)
            predictions = live_image_classifier.predict(input_fn=predict_input_fn)
        else:
            predictions = predict_fxn({"x": predict_data})
        
        # Visualization of the results of a detection
        # Detected objects correspond with the color of the outline box: 
        #   - green = tennis ball
        #   - orange = mango
        #   - yellow = lemon
        # Certainties/accuracies are printed out to the console on detection with the 
        # corresponding label
        if not use_saved_model:
            max_idx = np.argmax(predictions)
            predicted_certainty = predictions[max_idx]
        else:
            max_idx = np.argmax(predictions['scores'])
            predicted_certainty = predictions['scores'][max_idx]
        if max_idx == 0 and predicted_certainty > .4:
            cv2.rectangle(image_np, (0,0), (398,398), (0,255,0), 2)
            print("Detected Object: Tennis Ball; Certainty:", predicted_certainty * 100, "%")
        else if max_idx == 1 and predicted_certainty > .4:
            cv2.rectangle(image_np, (0,0), (398,398), (255,165,0), 2)
            print("Detected Object: Mango; Certainty:", predicted_certainty * 100, "%")
        else if predicted_certainty > .4:
            cv2.rectangle(image_np, (0,0), (398,398), (255,255,0), 2)
            print("Detected Object: Lemon; Certainty:", predicted_certainty * 100, "%")
        cv2.imshow('object detection', image_np)
        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
"""
    
if __name__ == "__main__":
  tf.app.run()
