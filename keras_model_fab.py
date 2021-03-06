# Deep learning lab course final project.
# Kaggle whale classification.

import os
import sys
import numpy as np
import time

#import h5py
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers

import utilities as ut
import keras_tools as tools

# Use pretrained model as described in https://keras.io/applications/

def _create_pretrained_model(config_dict, num_classes):
    #
    # extract relevant parts of configuration
    #
    num_dense_units_list = []
    base_model = config_dict['base_model']
    num_dense_layers = config_dict['num_dense_layers']
    for i in range(num_dense_layers):
        num_dense_units_list.append(config_dict['num_dense_units_' + str(i)])
    activation = config_dict['activation']
    optimizer = config_dict['optimizer']
    learning_rate = config_dict['learning_rate']

    #
    # load pre-trained model
    #
    if base_model == 'InceptionV3':
        pretrained_model = InceptionV3(weights='imagenet', include_top=False)  
    elif base_model == 'Xception':
        pretrained_model = Xception(weights='imagenet', include_top=False)
    elif base_model == 'ResNet50':
        pretrained_model = ResNet50(weights='imagenet', include_top=False)
    elif base_model == 'MobileNet':
        pretrained_model = MobileNet(weights='imagenet', input_shape=(224, 224,3), include_top=False)
    elif base_model == 'InceptionResNetV2':
        pretrained_model = InceptionResNetV2(weights='imagenet', include_top=False)
    else:
        print("invalid model: ", base_model)
    
    x = pretrained_model.output

    # for i, layer in enumerate(pretrained_model.layers):
    #    print(i, layer.name)    
    
    
    #
    # add fully connected layers
    #
    x = pretrained_model.output

    x = GlobalAveragePooling2D()(x)
    for i in range(num_dense_layers):
        x = Dense(num_dense_units_list[i], activation=activation)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    #
    # finish building combined model, lock parameters of pretrained part
    #
    model = Model(inputs=pretrained_model.input, outputs=predictions)
    for layer in pretrained_model.layers:
        layer.trainable = False
    if optimizer == 'SGD':
        opt = optimizers.SGD(lr=learning_rate)
    elif optimizer == 'Adam':
        opt = optimizers.Adam(lr=learning_rate)
    elif optimizer == 'RMSProp':
        opt = optimizers.RMSprop(lr=learning_rate)
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(optimizer))
    # compile the model (should be done *after* setting layers to
    # non-trainable)
    # metrics='accuracy' causes the model to store and report accuracy (train
    # and validate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.name = base_model    # to identify model in "unfreeze_layers() and "train()" function
    # print("successfuly created model: ", model.name)    
    
    return model


# "unfreeze_percentage" is fraction of whole CNN model to be unfrozen
# range 0.0 up to 0.3 - Values above 0.3 are interpreted as 0.3
def _unfreeze_cnn_layers(model, config_dict):
    # we chose to train the top X layers, where X is one of the nodes of the CNN
    # the first 2 layer blocks and unfreeze the rest:
    
    unfreeze_percentage = config_dict["unfreeze_percentage"]
    unfreeze_percentage = min(unfreeze_percentage, 0.3)  
    unfreeze_percentage = max(unfreeze_percentage, 0.0)  
    unfreeze_blocks = 0
    if model.name == 'InceptionV3':
        top_nodes = [280, 249, 229, 197]   # nodes of top layer-blocks: possible cut_off points
        unfreeze_blocks = int(11 * unfreeze_percentage)  # 
    elif model.name == 'Xception':
        top_nodes = [126, 116, 106, 96, 96]   
        unfreeze_blocks = int(12 * unfreeze_percentage)  # 
        if unfreeze_blocks > 0:
            cut_off = top_nodes[unfreeze_blocks-1]
    elif model.name == 'ResNet50':
        top_nodes = [161, 152, 140, 130, 120, 110]   
        unfreeze_blocks = int(16 * unfreeze_percentage)  # 
        if unfreeze_blocks > 0:
            cut_off = top_nodes[unfreeze_blocks-1]
    elif model.name == 'MobileNet':    # no nodes, cut_off after activations
        top_nodes = [79, 76, 73, 70, 67, 64, 61, 58, 55, 52]          
        unfreeze_blocks = int(27 * unfreeze_percentage)  # 
        if unfreeze_blocks > 0:
            cut_off = top_nodes[unfreeze_blocks-1]
    elif model.name == 'InceptionResNetV2':    # no nodes, cut_off after activations
        top_nodes = [761, 745, 729, 713, 697, 681, 665, 649, 633, 618, 594, 578, 562, 546]
        unfreeze_blocks = int(43 * unfreeze_percentage)  # 
        if unfreeze_blocks > 0:
            cut_off = top_nodes[unfreeze_blocks-1]                
    else:
        print("invalid model: ", model.name)

    if unfreeze_blocks > 0:
        
        cut_off = top_nodes[unfreeze_blocks-1]                
        for layer in model.layers[:cut_off]:
           layer.trainable = False
        for layer in model.layers[cut_off:]:
           layer.trainable = True        

        from keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=config_dict['cnn_learning_rate'], momentum=0.9), 
                      loss='categorical_crossentropy', metrics=['accuracy'])
        print("\n ****** {} unfrozen {} top blocks, cut_off after layer {} ******".format(model.name, unfreeze_blocks, cut_off-1))
    else:
        print("\n ****** {} no layers unfrozen".format(model.name))
        
    return model


def train(config_dict, 
          epochs,
          model=None,
          num_classes=10,
          save_model_path=None,
          save_data_path="plots",
          train_dir="data/model_train",
          valid_dir="data/model_valid",
          train_valid_split=0.7):

    start_time = time.time()
    #
    # extract relevant parts of configuration
    #
    cnn_unlock_epoch = config_dict["cnn_unlock_epoch"]
    unfreeze_percentage = config_dict["unfreeze_percentage"]
    batch_size = config_dict['batch_size']
    
    #
    # get model to train, determine training times
    #
    if model is None:
        model = _create_pretrained_model(config_dict, num_classes)
    
    if epochs <= cnn_unlock_epoch:
        training_epochs_dense = epochs
        training_epochs_wholemodel = 0
    else:
        training_epochs_dense = cnn_unlock_epoch
        training_epochs_wholemodel = epochs - cnn_unlock_epoch
    
    if model.name == 'InceptionV3' or model.name == 'Xception' or model.name == 'InceptionResNetV2':
        target_size = (299, 299)
    elif model.name == 'ResNet50' or model.name == 'MobileNet':
        target_size = (224, 224)
    else:
        print("invalid model: ", model.name)
    print("training model", model.name)    

    #
    # prepare training data
    #
    
    # create environment on filesystem with new random train/valid split
    num_train_imgs, num_valid_imgs = ut.create_small_case(
        sel_whales=np.arange(1, num_classes+1),
        train_dir=train_dir,
        valid_dir=valid_dir,
        train_valid=train_valid_split,
        sub_dirs=True)
    
    train_gen = image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale=1./255,   # redundant with featurewise_center ? 
        # preprocessing_function=preprocess_input, not used in most examples
        # horizontal_flip = True,    # no, as individual shapes are looked for
        fill_mode="nearest",
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=30)
    
    train_flow = train_gen.flow_from_directory(
        train_dir,
        # save_to_dir="data/model_train/augmented",    
        # color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size, 
        class_mode="categorical")
    
    valid_gen = image.ImageDataGenerator(
        rescale=1./255,
        fill_mode="nearest")
    
    valid_flow = valid_gen.flow_from_directory(
        valid_dir,
        target_size=target_size,
        class_mode="categorical") 

    #
    # train fully connected part
    #
    hist_dense = model.fit_generator(
        train_flow, 
        steps_per_epoch=num_train_imgs//batch_size,
        verbose=2, 
        validation_data=valid_flow,
        validation_steps=num_valid_imgs//batch_size,
        epochs=training_epochs_dense)
    histories = hist_dense.history
    #
    # train the whole model with parts of the cnn unlocked (fixed optimizer!)
    #
    if training_epochs_wholemodel > 0:
        model = _unfreeze_cnn_layers(model, config_dict)
        hist_wholemodel = model.fit_generator(
            train_flow, 
            steps_per_epoch = num_train_imgs//batch_size,
            verbose = 2, 
            validation_data = valid_flow,
            validation_steps = num_valid_imgs//batch_size,
            epochs=training_epochs_wholemodel)
        # concatenate training history
        for key in histories.keys():
            if type(histories[key]) == list:
                histories[key].extend(hist_wholemodel.history[key])
    
    #
    # do final cleanup
    #
    if save_model_path is not None:
        model.save(save_model_path)

    if save_data_path is not None:
        run_name = tools.get_run_name()
        tools.save_learning_curves(histories, run_name, base_path=save_data_path)
        csv_path = os.path.join(save_data_path, run_name, run_name + ".csv")
        ut.write_csv_dict(histories,
                          keys=['loss', 'acc', 'val_loss', 'val_acc'],
                          filename=csv_path)
        config_file_path = os.path.join(save_data_path, run_name, "config.txt")
        ut.append_to_file("configuration of this run:", config_file_path)
        ut.append_to_file(config_dict, config_file_path)
        ut.append_to_file("epochs=" + str(epochs), config_file_path)
        ut.append_to_file("num_classes=" + str(num_classes), config_file_path)
        ut.append_to_file("train_valid_split=" + str(train_valid_split), config_file_path)

    hpbandster_loss = 1.0 - histories['val_acc'][-1]
    runtime = time.time() - start_time
    return (hpbandster_loss, runtime, histories)


def eval_base_models(num_classes = 10):
    
    def eval_base_model(base_model, num_classes):

        config_dict = {'base_model': base_model, 
                       'num_dense_layers': 3,
                       'num_dense_units_0': 1024,
                       'num_dense_units_1': 1024,
                       'num_dense_units_2': 1024,
                       'activation': 'relu',
                       'optimizer': "RMSProp",
                       'learning_rate': 0.0001,
                       'cnn_learning_rate': 0.0001,               
                       'cnn_unlock_epoch': 2,
                       'unfreeze_percentage': 0.2,
                       'batch_size': 16}        
        
        model = _create_pretrained_model(config_dict, num_classes)
        _, _, history = train(config_dict, epochs=4, model=model,num_classes=num_classes)
        accs = history['val_acc']
        print("accs", accs)
        avg_acc = np.mean(history['val_acc'][-5:])
        MAP, _ = tools.print_model_test_info(model, num_classes)
        return (avg_acc, MAP)

    # base_models = ['InceptionV3', 'MobileNet', 'ResNet50']
    base_models = ['InceptionV3', 'MobileNet']
    results = []
    for base_model in base_models:
        results.append = eval_base_model(base_model, num_classes)

    print("results", results)
    
    # ut.save_bar_plot(results, base_models)
        
    return results
    
    
def main():
    print("****** Run short training with InceptionV3 and save results. ******")
    num_classes = 10
    config_dict = {'base_model': 'InceptionV3', 
                   'num_dense_layers': 3,
                   'num_dense_units_0': 500,
                   'num_dense_units_1': 250,
                   'num_dense_units_2': 50,
                   'activation': 'relu',
                   'optimizer': "SGD",
                   'learning_rate': 0.001,
                   'cnn_unlock_epoch': 8,
                   'unfreeze_percentage': 0.1,
                   'batch_size': 16}
    _, _, histories = train(config_dict, epochs=16, num_classes=num_classes)
    print("HISTORIES:")
    print(histories)
    run_name = tools.get_run_name()
    tools.save_learning_curves(histories, run_name)
    csv_path = os.path.join("plots/", run_name, "data.csv")
    ut.write_csv_dict(histories,
                      keys=['loss', 'acc', 'val_loss', 'val_acc'],
                      filename=csv_path)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
        exit()
    if "--class-graph" in sys.argv:
        tools.draw_num_classes_graphs()
        exit()
print("given command line options unknown.")
