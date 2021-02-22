import argparse
import numpy as np

import tensorflow as tf

from keras.models import load_model
from keras import optimizers, applications, backend
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from architecture import Architecture
from keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from keras.layers import BatchNormalization

def load_database(path_dataset, image_size, BS):

    train_datagen, val_datagen = data_generator()
    
    train_gen = train_datagen.flow_from_directory(
                                    path_dataset + "train/",
                                    target_size=(image_size[0], image_size[1]),
                                    batch_size=BS,
                                    class_mode="categorical",
                                    shuffle=True)
    val_gen = val_datagen.flow_from_directory(
                                    path_dataset + "val/",
                                    target_size=(image_size[0], image_size[1]),
                                    batch_size=BS,
                                    class_mode="categorical",
                                    shuffle=False)

    return train_gen, val_gen

def data_generator():
    print("[INFO] construct image generator for data augmentation of retrain model...")
    # construct the training image generator for data augmentation

    train_datagen = ImageDataGenerator(
                                #preprocessing_function=applications.resnet50.preprocess_input, # resnet50
                                #preprocessing_function=applications.xception.preprocess_input, # xception
                                preprocessing_function=applications.mobilenet.preprocess_input, # mobilenet
                                #preprocessing_function=applications.inception_v3.preprocess_input, # inception_v3
                                shear_range=0.3,
                                zoom_range=0.3, 
                                horizontal_flip=True)
    val_datagen = ImageDataGenerator(
                                #preprocessing_function=applications.resnet50.preprocess_input) # resnet50
                                #preprocessing_function=applications.xception.preprocess_input) # xception
                                preprocessing_function=applications.mobilenet.preprocess_input) # mobilenet
                                #preprocessing_function=applications.inception_v3.preprocess_input) # inception_v3
    
    return train_datagen, val_datagen

def callbacks_scheduler(path_save_model, plot_graph):
    # Callbacks for search learning rate and save best model
    my_callbacks = [ReduceLROnPlateau(
                                patience=2,
                                factor=0.2,
                                min_lr=0.0000000001,
                                verbose=1),
                    ModelCheckpoint(
                                filepath=path_save_model + plot_graph + "_soybean_disease_detection.h5",
                                monitor="val_loss",
                                verbose=0,
                                mode="auto",
                                save_freq="epoch",
                                save_best_only=False,
                                save_weights_only=False)]
    return my_callbacks



def load_saved_model(path_save_model, plot_graph):
    print ("[INFO] loading saved model")
    model = load_model(path_save_model + plot_graph + "_soybean_disease_detection.h5")

    return model

def change_learning_rate(model, LR):
    # To get learning rate
    print("[INFO] old Learning Rate: {}".format(backend.get_value(model.optimizer.lr)))

    # To set learning rate
    print("[INFO] seting new Learning Rate to: {}".format(LR))
    backend.set_value(model.optimizer.lr, LR)

    print("[INFO] new Learning Rate: {}".format(backend.get_value(model.optimizer.lr)))

    return model

def update_model(model):
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
    else:
        layer.trainable = True
    """
    for i, layer in enumerate(model.layers):
        print(i, layer.name, "-", layer.trainable)
    """
    return model

def retrain_saved_model(model, train_gen, val_gen, my_callbacks, BS, EPOCHS_old, EPOCHS_new):
    # Training the model
    print("[INFO] retraining the saved model...")
    history = model.fit(
                        train_gen,
                        validation_data=val_gen,
                        batch_size=BS,
                        epochs=EPOCHS_new,
                        shuffle=True,
                        initial_epoch=EPOCHS_old,
                        callbacks=[my_callbacks])

    return history

def run_evaluate_model(model, val_gen, BS):
    # make predictions on the validation set
    print("[INFO] evaluating network retrained into validation set with Classification report...")
    predictions = model.predict(val_gen, batch_size=BS)

    val_preds = np.argmax(predictions, axis=-1)
    
    # label names
    labels = list(val_gen.class_indices.keys())

    print ("Classification report : \n",classification_report(
                                val_gen.classes, val_preds,
                                target_names=labels))
    return classification_report

# Function to plot the learning curves
def plotLearningCurves(path_output, history, plot_graph):
    
    N = np.arange(0, len(history.history["val_loss"]))
    plt.style.use("seaborn")
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(212)
    plt.plot(N, history.history["accuracy"], label="train_acc")
    plt.plot(N, history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.savefig(path_output + plot_graph + "_retrained.pdf", bbox_inches='tight')

    return








if __name__ == '__main__':

    print ("[INFO] Starting retrain...")

    path_dataset = "/content/drive/MyDrive/databases/Soja_splited/"
    #plot_graph = "resnet50_loss_acc_512_1024_2048_drop05"
    plot_graph = "mobilenet_loss_acc_512_1024_2048_drop05"
    path_save_model = "../checkpoints/"
    path_output = "../output/"

    image_size = (224, 224, 3)
    BS = 128
    EPOCHS = 20
    LR = 1e-4
    EPOCHS_new = 50
    EPOCHS_old = EPOCHS

    model = load_saved_model(path_save_model, plot_graph)

    model = change_learning_rate(model, LR)

    model = update_model(model)

    train_gen, val_gen = load_database(path_dataset, image_size, BS)

    my_callbacks = callbacks_scheduler(path_save_model, plot_graph)

    #model = load_saved_model(path_save_model, plot_graph)
        
    history = retrain_saved_model(model, train_gen, val_gen, my_callbacks, BS, EPOCHS_old, EPOCHS_new)

    run_evaluate_model(model, val_gen, BS)

    plotLearningCurves(path_output, history, plot_graph)

    model.save(path_save_model + plot_graph + "_soybean_disease_detection_retrained.h5")
    
    print ("[INFO] OK retraining terminated...")

    # verificar como resetar o learning rate. esta muito pequeno




    



