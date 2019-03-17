import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import sys
import argparse
import numpy as np
import random
from model import *
from sklearn.metrics import accuracy_score
from batch import *
from test import *

torch.manual_seed(1000)
random.seed(1000)
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-batchSize", type = int, default = 10000)
    args.add_argument("-embeddings_type", type = str, default = "lstm")
    args.add_argument("-dataset", type = str, default = "verb_physics")
    args.add_argument("-task", type = str, default = "one_pole")
    args.add_argument("-dim", type = int, default = 1024)
    args.add_argument("-zero_shot", action="store_true")
    args.add_argument("-num_epochs", type = int, default = 100)
    args.add_argument("-dropout", type = float, default = 0.5)
    args.add_argument("-no_reverse", action="store_true")
    args.add_argument("-remove_NA", action="store_true")
    args.add_argument("-remove_sim", action="store_true")
    args.add_argument("-ignore_similar_emb", action="store_true")
    opts = args.parse_args()

    batch = Batch(opts.batchSize, opts.embeddings_type, opts.dataset, task=opts.task, dim = opts.dim, remove_NA=opts.remove_NA, remove_sim = opts.remove_sim)
    
    if opts.task == "three_way":
        model = PceThreeWay(opts.dim, dropout = opts.dropout)
    elif opts.task == "four_way":
        print("Using Four Way model")
        model = PceFourWay(opts.dim, dropout = opts.dropout, ignore_similar_emb = opts.ignore_similar_emb)
    elif opts.task == "one_pole":
        print("Using One Pole model")
        model = PceOnePole(opts.dim, dropout = opts.dropout)

    optim = Optim(model, lr = 0.01, weight_decay = 0) 
    epoch_num = 0

    if opts.dataset == "verb_physics":
        train_data = pickle.load(open("./orig/data/pickle2/verb_physics_train_5.pickle"))
        properties = []
        for i in train_data:
            properties.append(i[2])
        properties = list(set(properties))

    if not opts.zero_shot:
        while epoch_num < opts.num_epochs:
            input, target, epoch_end_flag = batch.next_batch()

            pred = model(input)
            optim.backward(pred, target)
            
            if epoch_end_flag:
                epoch_num += 1

            if epoch_num % 10 == 0 and epoch_end_flag:
                if epoch_num > 100:
                    optim.update_lr(optim.lr/1.05)
                else:
                    optim.update_lr(optim.lr/1.2)

            if epoch_num % 20 == 0 and epoch_end_flag and opts.dataset == "verb_physics":
                print("____________________________________________")
                for property in properties:
                    print("Epoch " + str(epoch_num) + ":\t" + property +":\t" +
                        str(round(
                            test(model, batch, zero_shot=True, zero_shot_property=property,batch_from="dev",no_reverse=opts.no_reverse),
                            2)) + ":" +   
                        str(round(
                            test(model, batch, zero_shot=True, zero_shot_property=property, batch_from="test", no_reverse=opts.no_reverse)
                            , 2))) 

                print("Epoch " + str(epoch_num) + ":overall:" + 
                          str(test(model, batch, batch_from="dev", no_reverse=opts.no_reverse )) + ":" + 
                          str(test(model, batch, batch_from="test", no_reverse=opts.no_reverse )))
                pickle.dump(model, open("model.pkl", "wb"))

            elif epoch_num % 20 == 0 and epoch_end_flag and opts.dataset == "PCE":
                print("Epoch " + str(epoch_num) + " : " + str(test(model, batch)))

    if opts.zero_shot:
        for property in properties:
            
            #Reset Model
            if opts.task == "three_way":
                model = PceThreeWay(opts.dim, dropout = opts.dropout)
            elif opts.task == "four_way":
                print("Using Four Way model")
                model = PceFourWay(opts.dim, dropout = opts.dropout, ignore_similar_emb = opts.ignore_similar_emb)
            elif opts.task == "one_pole":
                print("Using One Pole model")
                model = PceOnePole(opts.dim, dropout = opts.dropout)



            print("____________________________________________")
            print("PROPERTY: " + property)
            epoch_num = 0
            while epoch_num < opts.num_epochs:
                input, target, epoch_end_flag = batch.next_batch(zero_shot=True, zero_shot_property=property)

                # print("TRAIN: batch size: " + str(len(input)))
                pred = model(input)
                optim.backward(pred, target)
                
                if epoch_end_flag:
                    epoch_num += 1

                if epoch_num > 100:
                    optim.update_lr(optim.lr/1.05)
                else:
                    optim.update_lr(optim.lr/1.2)

                if epoch_num % 10 == 0 and epoch_end_flag:
                    print("Epoch " + str(epoch_num) + ":" + 
                            str(round(test(model, batch, zero_shot=True, zero_shot_property=property,batch_from="dev"), 3)) + ":" +   
                            str(round(test(model, batch, zero_shot=True, zero_shot_property=property, batch_from="test"), 3))) 


























