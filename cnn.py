# Deep learning lab course final project.  Kaggle whale
# classification.

# Build a tensorflow model according to the hyperparameters provided.

import tensorflow as tf
import csv


def extract_labels(path="data/train.csv"):
    """Extract image labels from csv and return as {file_name: whale_name}
dict."""
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        mydict = {rows[0]:rows[1] for rows in reader}
    return mydict


def build_model(features, labels, mode, params):
    """Returns an EstimatorSpec for the model described by the
hyperparameters. This is the model_fn as in
https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier

    """
    # layers_list is a list of the form (pool/conv/dense, filters,
    # size, units, activation)

    # unpack parameters
    layers_list = params["layers_list"]
    optimizer = params["optimizer"]
    optimizer_params = params["optimizer_params"]
    n_classes = params["n_classes"]
    image_x = params["image_x"]
    image_y = params["image_y"]
    
    input_layer = tf.reshape(features, [-1, image_x, image_y, 1])
    # labels = tf.placeholder(tf.float32)

    layers = [input_layer]  # tf layers
    # add layers according to the descriptions 
    for description in layers_list:
        type, filters, size, units, activation = description
        if type == "pool":
            layer = tf.layers.max_pooling2d(inputs=layers[-1],
                                            pool_size=[size, size],
                                            strides=size,
                                            padding="valid")
        elif type == "conv":
            layer = tf.layers.conv2d(inputs=layers[-1],
                                     filters=filters,
                                     kernel_size=[size, size],
                                     padding="same",
                                     activation=activation)
        elif type == "dense":
            layer = tf.layers.dense(inputs=layers[-1],
                                    units=units,
                                    activation=activation)
        elif type == "flatten":
            layer = tf.contrib.layers.flatten(inputs=layers[-1])

        layers.append(layer)

    # add softmax output layer
    output = tf.layers.dense(inputs=layers[-1], units=n_classes)
    classes = tf.argmax(input=output, axis=1)
    softmax = tf.nn.softmax(output, name="softmax_output")

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
        "classes": classes,
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": softmax
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # one-hot encoding and cross entropy loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=n_classes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=output)    

    correct_prediction = tf.equal(classes, tf.cast(labels, dtype=tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        if optimizer == "SGD":
            learning_rate, = optimizer_params
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = { "accuracy":
                            tf.metrics.accuracy(labels=labels,
                                                predictions=classes)}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
