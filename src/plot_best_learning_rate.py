#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import sys
import matplotlib
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import argparse

# reload(sys)  
# sys.setdefaultencoding('utf8')

def get_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--project", type=str, required=True,
        help="path of the project. Normally KittiSeg/RUNS/...")
    ap.add_argument("-n", "--name", type=str, required=True,
        help="path of the experiment name. Normally KiitiSeg/RUNS/project/")
    ap.add_argument("-k0", "--kfold_0", type=str, required=False,
        help="Experiment name of the [0] k-fold")
    ap.add_argument("-k1", "--kfold_1", type=str, required=False,
        help="Experiment name of the [1] k-fold")
    ap.add_argument("-k2", "--kfold_2", type=str, required=False,
        help="Experiment name of the [2] k-fold")
    ap.add_argument("-k3", "--kfold_3", type=str, required=False,
        help="Experiment name of the [3] k-fold")
    ap.add_argument("-k4", "--kfold_4", type=str, required=False,
        help="Experiment name of the [4] k-fold")    
    ap.add_argument("-evi", "--evalIters", type=int, default=400,
        help="Number of iterations for each evaluation. Obs: eval_iters parameter")
    ap.add_argument("-mi", "--maxIters", type=int, default=36000,
        help="Number of total iterations for training. Obs: max_iters parameter")
    args = vars(ap.parse_args())

    return args["project"], args["name"], args["kfold_0"], args["kfold_1"], args["kfold_2"], args["kfold_3"], args["kfold_4"], args["evalIters"], args["maxIters"]

#eval_iters = 400
#max_iters = 36000

def plotGraphLR(project, name, kfold_0,  kfold_1, kfold_2, kfold_3, kfold_4, eval_iters, max_iters):

    label_list = xrange(eval_iters, max_iters+1, eval_iters)

    plt.figure(figsize=(6, 3))
    #plt.title(u'Avaliação dos classificadores (Métrica Fmax)')
    plt.rcParams.update({'font.size': 6})

    listaLog = []    
    #for i in xrange(1):
        #listaLog.append("/home/csbezerra/KittiSeg/RUNS/"+project+"/"+name+"/"+"output.log")

    listaLog.append("/home/cides/logs_nice1/output_0.log")
    listaLog.append("/home/cides/logs_nice1/output_1.log")
    listaLog.append("/home/cides/logs_nice1/output_2.log")
    listaLog.append("/home/cides/logs_nice1/output_3.log")
    listaLog.append("/home/cides/logs_nice1/output_4.log")
    
    #listaLog.append("/output.log")

    listaCor = ['blue', 'red', 'green', 'pink', 'cyan']
    linestyles = ['--', '--', '--', '--', '--']
    markers = ['v', '*', '8', 's', 'p']

    label = []
    for indice in xrange(len(listaLog)):
        #label.append('Train. clas. '+str(indice+1))
        label.append('Fscore (raw), k = '+ str(indice))
        listDataValRaw = read_values(listaLog[indice], 'val', 'Fscore', 'raw')
        plt.plot(label_list, listDataValRaw, marker=markers[indice], markersize=2.5, markeredgewidth=0.0, color=listaCor[indice], linestyle=' ')
        label.append('Fscore (smooth), k = '+ str(indice))
        listDataValSmooth = read_values(listaLog[indice], 'val', 'Fscore', 'smooth')
        plt.plot(label_list, listDataValSmooth, marker="", color=listaCor[indice], linestyle='-', lw=0.7)
        print("K-fold: {} | Best Weight: {} | Iteration index: {} (Raw)".format(indice+1, max(listDataValRaw), (eval_iters*(np.argmax(listDataValRaw)+1))))
        print("K-fold: {} | Best Weight: {} | Iteration index: {} (Smoothed)".format(indice+1, max(listDataValSmooth), (eval_iters*(np.argmax(listDataValSmooth)+1))))

    #plt.yticks(np.array(range(0, 81, 10)))
    #plt.legend(label, ncol=2, loc="lower right", columnspacing=0.5, labelspacing=0.0, handletextpad=0.0, handlelength=1.0, fancybox=True, shadow=True, prop={'size': 12})
    #plt.legend(label, ncol=5, bbox_to_anchor=(0.15, 0.15, 0.85, 0.0), columnspacing=0.5, labelspacing=0.0, handletextpad=0.0, handlelength=1.0, fancybox=True, shadow=True, prop={'size': 12})
    plt.legend(label, loc="lower right", shadow = True, fancybox=True)
    plt.xlabel('Iteration') 
    plt.ylabel('Validation Score [%]')
    plt.grid(linestyle=':', lw=0.2)
    
    # Colocar os nomes e project pra salvar junto
    plt.savefig("../"+ project +"_fscore_5_kfolds.eps", bbox_inches='tight')

    return

def read_values_learning_rate(filename, str_lr):
    regex_string = "\s+\%s\s+:\s+(\d+\.\d+)" % (str_lr)
    
    regex = re.compile(regex_string)

    value_list = [regex.search(line).group(1) for line in open(filename) if regex.search(line) is not None]

    float_list = [float(value) for value in value_list]

    return float_list

def read_values_optimizers(filename, str_optimizer):
    regex_string = "\s+\%s\s+:\s+(.*)" % (str_optimizer)
    
    regex = re.compile(regex_string)

    value_list = [regex.search(line).group(1) for line in open(filename) if regex.search(line) is not None]

    str_list = [str(value) for value in value_list]

    return str_list

def read_values_epocs(filename, epoch):
    
    regex_string = "\%s\s+(\d+)[^\d+:]" % (epoch)
    
    regex = re.compile(regex_string)

    value_list = [regex.search(line).group(1) for line in open(filename) if regex.search(line) is not None]

    int_list = [int(value) for value in value_list]

    return int_list

def read_values_metrics(filename, metrics_names):
    
    dict_metrics = {}
    dict_metrics = {
                "loss" : [],
                "accuracy" : [],
                "val_loss" : [],
                "val_accuracy" : []
        }
    

    for metric_name in metrics_names:
        regex_string = "\s%s\:\s(\d+\.\d+)" % (metric_name)
        regex = re.compile(regex_string)
        
        float_list = []
        for line in open(filename):
            if regex.search(line) is not None:
                float_list.append(float(regex.search(line).group(1)))
        dict_metrics[metric_name] = float_list
        
        # float_list = [float(value) for value in value_list]
    
    

    # value_list = [regex.search(line).group(1) for line in open(filename) if regex.search(line) is not None]

    # float_list = [float(value) for value in value_list]

    return dict_metrics

def read_values(filename):
    str_lr = "Learning_rate"
    str_optimizer = "Optimizer"
    epoch = "Epoch"
    metrics_names = ["loss", "accuracy", "val_loss", "val_accuracy"]
    dict_log_values = {
            "Learning_rate" : [],
            "Optimizer" : [],
            "Epoch" : [],
            "loss" : [],
            "accuracy" : [],
            "val_loss" : [],
            "val_accuracy" : []
            }
    
    regex_str_lr = "\s+\%s\s+:\s+(\d+\.\d+)" % (str_lr)    
    regex_lr = re.compile(regex_str_lr)
    
    regex_str_optimizer = "\s+\%s\s+:\s+(.*)" % (str_optimizer)
    regex_optimizer = re.compile(regex_str_optimizer)
    
    regex_str_epoch = "\%s\s+(\d+)[^\d+:]" % (epoch)
    regex_epoch = re.compile(regex_str_epoch)
    
    regex_str_loss = "\s%s\:\s(\d+\.\d+)" % (metrics_names[0])
    regex_loss = re.compile(regex_str_loss)
    
    regex_str_accuracy = "\s%s\:\s(\d+\.\d+)" % ("accuracy")
    regex_accuracy = re.compile(regex_str_accuracy)
    
    regex_str_val_loss = "\s%s\:\s(\d+\.\d+)" % ("val_loss")
    regex_val_loss = re.compile(regex_str_val_loss)
    
    regex_str_val_accuracy = "\s%s\:\s(\d+\.\d+)" % ("val_accuracy")
    regex_val_accuracy = re.compile(regex_str_val_accuracy)
    
    regex_str_all_metrics = "\s%s\:\s(\d+\.\d+)\s+\-\s%s\:\s(\d+\.\d+)\s+\-\s%s\:\s(\d+\.\d+)\s+\-\s%s\:\s(\d+\.\d+)" % (metrics_names[0], metrics_names[1], metrics_names[2], metrics_names[3])
    regex_all_metrics = re.compile(regex_str_all_metrics)
    
    list_lr = []
    list_optimizer = []
    list_epoch = []
    list_loss = []
    list_accuracy = []
    list_val_loss = []
    list_val_accuracy = []
    list_all_metrics = []
    
    for line in open(filename):
        # print (line)
        if regex_lr.search(line) is not None:
            list_lr.append(float(regex_lr.search(line).group(1)))
        elif regex_optimizer.search(line) is not None:
            list_optimizer.append(str(regex_optimizer.search(line).group(1)))
        elif regex_epoch.search(line) is not None:
            list_epoch.append(int(regex_epoch.search(line).group(1)))
        # elif regex_loss.search(line) is not None:
            # list_loss.append(float(regex_loss.search(line).group(1)))
        # elif regex_accuracy.search(line) is not None:
            # list_accuracy.append(float(regex_accuracy.search(line).group(1)))
        # elif regex_val_loss.search(line) is not None:
            # list_val_loss.append(float(regex_val_loss.search(line).group(1)))
        # elif regex_val_accuracy.search(line) is not None:
            # list_val_accuracy.append(float(regex_val_accuracy.search(line).group(1)))
        elif regex_all_metrics.search(line) is not None:
            metric = regex_all_metrics.search(line).group(1, 2, 3, 4)
            list_loss.append(float(metric[0]))
            list_accuracy.append(float(metric[1]))
            list_val_loss.append(float(metric[2]))
            list_val_accuracy.append(float(metric[3]))
            # list_all_metrics.append(regex_all_metrics.search(line).group(1, 2, 3, 4))
            
            
    dict_log_values["Learning_rate"] = list_lr
    dict_log_values["Optimizer"] = list_optimizer
    dict_log_values["Epoch"] = list_epoch
    dict_log_values["loss"] = list_loss
    dict_log_values["accuracy"] = list_accuracy
    dict_log_values["val_loss"] = list_val_loss
    dict_log_values["val_accuracy"] = list_val_accuracy
            
            
            # list_all_metrics.append(float(regex_all_metrics.search(line).group(1, 2, 3, 4)))
    # list_all_metrics = np.array(list_all_metrics, dtype=np.float32)
    
    # print (list_lr)
    # print (list_optimizer)
    # print (list_epoch)
    # print (list_loss)
    # print (list_accuracy)
    # print (list_val_loss)
    # print (list_val_accuracy)
    # print (list_all_metrics)#.astype(float)
    # print (dict_log_values)
    
    return dict_log_values

# Function to plot the learning curves
def plotLearningCurves(name, dict_log_values):
    # plot the training loss and accuracy
    #N = np.arange(0, EPOCHS)
    # for lr in learning_rates:
        # for opt in optimizers:
    
    learning_rates = [1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1E-0]
    optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

    # N = np.arange(0, 344)
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 8))
    # plt.ylim(0, 1) # set the vertical range to [0-1]
    plt.plot(dict_log_values["Learning_rate"], dict_log_values["accuracy"], label="train_acc")
    plt.plot(dict_log_values["Learning_rate"], dict_log_values["val_accuracy"], label="val_acc")
    plt.title("Training Accuracy Search Learning Rate and Optimizers")
    # plt.xlabel("Epoch #")
    plt.xlabel(learning_rates)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("./output/accuracy.pdf")
    
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 8))
    # plt.ylim(0, 1) # set the vertical range to [0-1]
    plt.plot(dict_log_values["Learning_rate"], dict_log_values["loss"], label="train_loss")
    plt.plot(dict_log_values["Learning_rate"], dict_log_values["val_loss"], label="val_loss")
    plt.title("Training Loss Search Learning Rate and Optimizers")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./output/loss.pdf")

    return
    

def main():
    # get parameters
    #project, name, kfold_0,  kfold_1, kfold_2, kfold_3, kfold_4, eval_iters, max_iters = get_args()
    
    filename = "/home/csbezerra/DeepLearning/soybean-diseases/output_vgg16_64.log"
    # name = "/home/csbezerra/DeepLearning/soybean-diseases/kittiSeg_output_0.log"
    
    # # float_list = read_values(name, 'val', 'Fscore', 'raw')
    # learning_rates = read_values_learning_rate(name, 'Learning_rate')
    # print (learning_rates)
    
    # optimizers = read_values_optimizers(name, 'Optimizer')
    # print (optimizers)
    
    # epochs = read_values_epocs(name, 'Epoch')
    # print (epochs)
    
    # metrics_names = ["loss", "accuracy", "val_loss", "val_accuracy"]
    
    # dict_metrics = read_values_metrics(name, metrics_names)
    
    # print (np.shape(learning_rates))
    # print (np.shape(optimizers))
    # print (np.shape(epochs))
    # print (len(dict_metrics[metrics_names[0]]))
    
    dict_log_values = read_values(filename)
    
    print (np.shape(dict_log_values["Epoch"]))
    
    # for lr in dict_log_values["Learning_rate"]:
        # print (lr)
    # for opt in range (0, 7):
        # for lr in range (0, len(dict_log_values["Learning_rate"]), 7):
            # print (dict_log_values["Optimizer"][opt], dict_log_values["Learning_rate"][lr])
    
    
    for i, epc in enumerate(dict_log_values["Epoch"]):
        m = 0
        # print (i, epc)
        if i <= epc:
            m = i
            # print ("M antes: ", m)
            print (dict_log_values["loss"][m], i)
        # print (i, epc)
        else:
            m = m + i
            # print ("M depois: ", m)
            print (dict_log_values["loss"][m], i)
        
        # print (i, epc)
    
    plotLearningCurves(filename, dict_log_values)
    
    # print (dict_log_values["loss"][])
    
    # plotGraphLR(name)








if __name__ == '__main__':
    main()
