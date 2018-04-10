# Deep learning lab course final project.
# Kaggle whale classification.

# Helper functions for the main keras model.

import datetime
import os
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import keras.utils
import utilities as ut


def get_run_name(prefix="run", additional=""):
    return "_".join([prefix, 
                     datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"),
                     additional])


def save_learning_curves(history, run_name, base_path="plots/"):
    """Saves the data from keras history dict in loss and accuracy graphs to folder
    specified by base_path and run_name."""
    path = os.path.join(base_path, run_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    losses = {k: history[k] for k in ['loss', 'val_loss']}
    accuracies = {k: history[k] for k in ['acc', 'val_acc']}
    x = range(len(losses['loss']))
    fn_losses = os.path.join(path, "loss.png")
    fn_accuracies = os.path.join(path, "accuracy.png")
    ut.save_plot(x, ys=losses, xlabel="epoch", ylabel="loss",
                 title=run_name, path=fn_losses)
    ut.save_plot(x, ys=accuracies, xlabel="epoch", ylabel="accuracy",
                 title=run_name, path=fn_accuracies)


def save_learning_curves_2(history, cnn_after, run_name, base_path="plots/"):
    """Saves the data from keras history dict in loss and accuracy graphs to folder
    specified by base_path and run_name."""
    path = os.path.join(base_path, run_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    # losses = {k: history[k] for k in ['loss', 'val_loss']}
    accuracies = {k: history[k] for k in ['val_acc','acc']}
    accuracies = {k: history[k] for k in ['val_acc']}
    
    x = range(len(accuracies['val_acc']))
    # fn_losses = os.path.join(path, "loss.png")
    fn_accuracies = os.path.join(path, "accuracy.png")
    ut.save_plot_2(cnn_after, x, ys=accuracies, xlabel="epoch", ylabel="accuracy",
                 title=run_name, path=fn_accuracies)    
    

def draw_num_classes_graphs():
    print("Will likely not work because")
    print("keras_tools.draw_num_classes_graphs() was not yet adapted")
    print("to the usage of config_dict in keras_model.py")
    """Train network and save learning curves for different values for num_classes."""
    values = [10, 50, 100, 250, 1000, 4000]
    for num_classes in values:
        print("Training model on {} most common classes.".format(num_classes))
        model = create_pretrained_model(num_classes=num_classes)
        histories = train(model, num_classes, epochs=50)
        run_name = get_run_name("{}classes".format(num_classes))
        save_learning_curves(histories, run_name)
        csv_path = os.path.join("plots/", run_name, "data.csv")
        ut.write_csv_dict(histories,
                       keys=['loss', 'acc', 'val_loss', 'val_acc'],
                       filename=csv_path)


def visualize_model(model=None, 
                    filename="InceptionV3_visualization.png",
                    show_shapes=False):
    """
    Write graph visualization of Keras Model to file.
    Default model is InceptionV3
    """
    if model is None:
        model = InceptionV3(weights='imagenet', include_top=False)
    else:
        model = model
    keras.utils.print_summary(model)
    print("---")
    print("len(model.layers)", len(model.layers))
    print("saveing graph visualization to file")
    keras.utils.plot_model(model, show_shapes=show_shapes, to_file=filename)
    print("saved graph visualization to file")

    
def compute_preds(model, num_classes, train_dir = "data/model_train", 
                  test_dir = "data/model_valid", test_csv = "data/model_valid.csv"):
    
    batch_size = 16     # used for training as well as validation
    max_preds = 5        # number of ranked predictions (default 5)
    
    if model.name == 'InceptionV3' or model.name == 'Xception' or model.name == 'InceptionResNetV2':
        target_size = (299, 299)
    elif model.name == 'ResNet50' or model.name == 'MobileNet':
        target_size = (224, 224)
    else:
        print("invalid model: ", model.name)
    print("training model", model.name) 
    
    '''    
    num_train_imgs, num_valid_imgs = ut.create_small_case(
        sel_whales = np.arange(1,num_classes+1),  # whales to be considered
        all_train_dir = all_train_dir,
        all_train_csv = all_train_csv,
        train_dir = test_dir,
        train_csv = test_csv,
        valid_dir = None,     # no validation, copy all data into test_dir "data/model_test"
        valid_csv = None,
        train_valid = 1.,
        sub_dirs = True) 
    '''
    test_gen = image.ImageDataGenerator(
        rescale = 1./255,
        fill_mode = "nearest")

    test_flow = test_gen.flow_from_directory(
        test_dir,
        shuffle=False,          
        batch_size = batch_size,     
        target_size = target_size,
        class_mode = None)    # use "categorical" ??
    
    preds = model.predict_generator(test_flow, verbose = 1)

    # whale_class_map = (test_flow.class_indices)           # get dict mapping whalenames --> class_no
    class_whale_map = ut.make_label_dict(directory=train_dir) # get dict mapping class_no --> whalenames
    '''
    print("whale_class_map:")
    print(whale_class_map)
    print("class_whale_map:")
    print(class_whale_map)
    print("preds.shape:")
    print(preds.shape)
    print("preds[:10]")
    print(preds[:10])
    '''
    # get list of model predictions: one ordered list of maxpred whalenames per image
    top_k = preds.argsort()[:, -max_preds:][:, ::-1]
    model_preds = [([class_whale_map[i] for i in line]) for line in top_k]  
    
    # get list of true labels: one whalename per image
    true_labels = []
    file_names = []
    if test_csv != '':
        test_list = ut.read_csv(file_name = test_csv)    # list with (filename, whalename)
    i = 0    
    for fn in test_flow.filenames:
        if i<3:
            print("fn",fn)
        i=i+1
        offset, directory, filename = fn.split('/')
        file_names.append(filename)
        if test_csv != '':
            whale = [line[1] for line in test_list if line[0]==filename][0]
            true_labels.append(whale)    

    return file_names, model_preds, true_labels

   
    
def write_pred_to_csv(file_names, model_preds, path = "data/submission.csv"):
    csv_list = []
    for i in range(len(model_preds)):
        csv_row = ['','']
        csv_row[0] = file_names[i]
        s = 'new_whale'    # string containing the five whale names separated by blanks
        for j in range(len(model_preds[i])-1):   # run over 5 ordered predictions
            # if j>0:
            s = s + ' '
            s = s + model_preds[i][j]
            # print("next_s", s)
        csv_row[1] = s
        csv_list.append(csv_row)
    # print("csv_list", csv_list)
    print("write csv file")
    ut.write_csv(csv_list, path)
    print("done writing csv file")      


# perform prediction on validation data, compare with true labels and compute acc and MAP    
def compute_map(model_preds, true_labels):
    max_preds = len(model_preds[0])
    print("max_preds", max_preds)
    # print("model predictions: \n", np.array(model_preds)[0:10])
    # print("true labels \n", np.array(true_labels)[0:10])
    
    # compute accuracy by hand
    TP_List = [(1 if model_preds[i][0]==true_labels[i] else 0) for i in range(len(true_labels))]
    acc = np.sum(TP_List) / len(true_labels)
    print("{} true predictions out of {}: accurracy: {} ".format(np.sum(TP_List),len(true_labels),acc))

    MAP = ut.mean_average_precision(model_preds, true_labels, max_preds)
    print("MAP", MAP)
    
    return MAP



if __name__ == "__main__":
    import sys
    if "--visualize_inceptionV3" in sys.argv:
        visualize_model()
