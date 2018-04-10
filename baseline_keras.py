# Deep learning lab course final project.
# Kaggle whale classification.

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# from Dataloader import load_train_batch
import os
import skimage.io
import numpy as np
from utilities import write_csv


INPUT_DIRECTORY = "baseline/"
OUTPUT_FILE = "submission.csv"


# generate image batches, replace by Keras ImageDataGenerator
# def input_generator(batch_size=64, directory="./data/test"):
#     i = 0
#     images = []
#     for filename in os.listdir(directory):
#         if i == batch_size:
#             i = 0
#             yield np.array(images)
#             images = []
#         images.append(skimage.io.imread(os.path.join(directory, filename)))
#         i += 1


# Use pretrained model as described in https://keras.io/applications/


def make_label_dict(directory="baseline/"):
    label_dict = dict()
    for i, label in enumerate(sorted(os.listdir(directory))):
        label_dict[i] = label
    return label_dict


def main():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    num_classes = len(os.listdir(INPUT_DIRECTORY))
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # define image generator
    train_gen = image.ImageDataGenerator()
    
    # train the model on the new data for a few epochs
    model.fit_generator(train_gen.flow_from_directory(INPUT_DIRECTORY),
                        steps_per_epoch=3, epochs=1, verbose=2)

    # let's predict the test set to see a rough score
    labels = make_label_dict()
    test_gen = image.ImageDataGenerator()
    flow = test_gen.flow_from_directory(INPUT_DIRECTORY, class_mode=None)
    predictions = model.predict_generator(flow, verbose=1) # steps=15611//32)
    top_k = predictions.argsort()[:, -4:][:, ::-1]
    classes = [" ".join([labels[i] for i in line]) for line in top_k]
    filenames = flow.filenames  # [os.path.basename(f) for f in flow.filenames]
    csv_list = zip(filenames, classes)
    write_csv(csv_list, file_name=OUTPUT_FILE)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    #for i, layer in enumerate(base_model.layers):
    #   print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    #for layer in model.layers[:249]:
    #   layer.trainable = False
    #for layer in model.layers[249:]:
    #   layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    #from keras.optimizers import SGD
    #model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    #model.fit_generator(...)


main()

