import argparse
import numpy as np

import tensorflow as tf

from keras import optimizers, applications
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from architecture import Architecture
from keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-dt", "--dataset", required=True,
        help="path to input dataset")
    ap.add_argument("-ou", "--output", required=True,
        help="path to output")
    ap.add_argument("-p", "--plot", type=str, default="loss_accuracy.pdf",
        help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str, default="soybean_disease_detection.model",
        help="path to output soybean disease detector model")
    args = vars(ap.parse_args())
    
    return args["dataset"], args["output"], args["plot"], args["model"]

def load_database(path_dataset, image_size, BS):

    train_datagen, val_datagen = data_generator()
    
    train_gen = train_datagen.flow_from_directory(
                                    path_dataset + "train/",
                                    target_size=(image_size[0], image_size[1]),
                                    batch_size=BS,
                                    class_mode="categorical",
                                    shuffle=True)#,
                                    #subset="training")
    val_gen = val_datagen.flow_from_directory(
                                    path_dataset + "val/",
                                    target_size=(image_size[0], image_size[1]),
                                    batch_size=BS,
                                    class_mode="categorical",
                                    shuffle=False)#,
                                    #subset="validation")

    return train_gen, val_gen

def data_generator():
    print("[INFO] construct image generator for data augmentation...")
    # construct the training image generator for data augmentation
    '''
    train_datagen = ImageDataGenerator(
                                featurewise_center=True,
                                rotation_range=40,
                                zoom_range=0.25,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.15,
                                horizontal_flip=True,
                                fill_mode="nearest")#,
                                #validation_split=0.2)
                                '''
    #train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)

    train_datagen = ImageDataGenerator(
                                preprocessing_function=applications.resnet50.preprocess_input,
                                #featurewise_center=True,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
    val_datagen = ImageDataGenerator(
                                preprocessing_function=applications.resnet50.preprocess_input)#featurewise_center=True)

    # Normalize data with mean values from imagenet dataset
    train_datagen.mean = [123.68, 116.779, 103.939]
    val_datagen.mean = [123.68, 116.779, 103.939]
    
    return train_datagen, val_datagen

def callbacks_scheduler(path_output):
    # Callbacks for search learning rate and save best model
    my_callbacks = [
                    ReduceLROnPlateau(
                                patience=2,
                                factor=0.3,
                                min_lr=0.0000000001,
                                verbose=1),
                    ModelCheckpoint(
                                filepath=path_output + "soybean_disease_detection.h5",
                                monitor="val_accuracy",
                                mode="max",
                                save_best_only=True)]
    return my_callbacks

# Function to plot the learning curves
def plotLearningCurves(path_output, history, plot_graph, result):
    
    N = np.arange(0, len(history.history["val_loss"]))
    plt.style.use("seaborn")
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss" + " {:.2f}".format(result[0]))
    plt.title("Training Loss and Accuracy")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(212)
    plt.plot(N, history.history["accuracy"], label="train_acc")
    plt.plot(N, history.history["val_accuracy"], label="val_acc" + " {:.2f}%".format(result[1] * 100.0))
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.savefig(path_output + plot_graph + ".pdf", bbox_inches='tight')

    return

def create_run_train_model(train_gen, val_gen, path_save_model, image_size, my_callbacks, LR, BS, EPOCHS):
    
    model = Architecture.resnet50_transfer_learning(image_size)

    #opt = optimizers.SGD(learning_rate=LR, momentum=0.9, decay=LR / EPOCHS)
    #opt = optimizers.Adam(learning_rate=LR)
    opt = optimizers.RMSprop(learning_rate=LR)

    
    #model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]) # To binary class problem
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    #model.summary()
    plot_model(model, to_file=path_save_model + "model_plot.png", show_shapes=True, show_layer_names=True)

    # Training the model
    print("[INFO] training the model with data augmentation...")
    history = model.fit(
                        train_gen,
                        validation_data=val_gen,
                        batch_size=BS,
                        epochs=EPOCHS,
                        shuffle=True,
                        callbacks=[my_callbacks])
    return history, model

def run_evaluate_model(model, val_gen, BS):
    # make evaluation on the validation set
    print("[INFO] evaluating network into validation set...")
    result = model.evaluate(val_gen)

    predictions = model.predict(val_gen, batch_size=BS)
    
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predictions = np.argmax(predictions, axis=1)

    return result, predictions

def run_evaluate_model_2(model, val_gen, BS):
    # make predictions on the validation set
    print("[INFO] evaluating network into validation set...")
    predictions = model.predict(val_gen, batch_size=BS)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    val_preds = np.argmax(predictions, axis=-1)
    
    #val_trues = val_gen.classes
    #print ("val_gen.classes: ", val_gen.classes)

    # label names
    labels = list(val_gen.class_indices.keys())
    #print ("Label Names: ", labels)

    # show a nicely formatted classification report
    #print ("Classification report : \n",classification_report(
    #                    val_labels.argmax(axis=1), predIdxs,
    #                    target_names=class_names))
    print ("Classification report : \n",classification_report(
                                val_gen.classes, val_preds,
                                target_names=labels))
    return


if __name__ == '__main__':
    
    path_dataset, path_output, plot_graph, path_save_model = get_args()
    print ("[INFO] Starting...")

    image_size = (224, 224, 3)
    BS = 128
    EPOCHS = 15
    LR = 1e-5
    
    train_gen, val_gen = load_database(path_dataset, image_size, BS)

    print(train_gen.class_indices)
    #print(val_gen.class_indices)

    #print ("val_gen.classes: ", val_gen.classes)
    
    my_callbacks = callbacks_scheduler(path_output)
    
    history, model = create_run_train_model(train_gen, val_gen, path_save_model, image_size, my_callbacks, LR, BS, EPOCHS)
    
    result, predictions = run_evaluate_model(model, val_gen, BS)

    run_evaluate_model_2(model, val_gen, BS)

    print ("Predictions: {}".format(predictions))

    print ("Validation Loss: {:.2f}".format(result[0]))
    print ("Validation Accuracy: {:.2f}%".format(result[1] * 100.0))
    
    plotLearningCurves(path_output, history, plot_graph, result)

    model.save(path_save_model + "soybean_disease_detection.h5")
    
    print ("[INFO] OK training terminated...")
    
    
    
    
    
    
    
    
