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
    args.add_argument("-batchSize", type = int, default = 100)
    args.add_argument("-embeddings_type", type = str, default = "word2vec")
    args.add_argument("-dataset", type = str, default = "verb_physics")
    args.add_argument("-task", type = str, default = "three_way")
    args.add_argument("-dim", type = int, default = 300)
    opts = args.parse_args()

    print("Loading Training data from " + opts.dataset)
    batch = Batch(opts.batchSize, opts.embeddings_type, opts.dataset, task=opts.task, dim = opts.dim)
    
    if opts.task == "three_way":
        model = PceThreeWay(opts.dim)
    elif opts.task == "four_way":
        print("Using Four Way model")
        model = PceFourWay(opts.dim)
    elif opts.task == "one_pole":
        print("Using One Pole model")
        model = PceOnePole(opts.dim)

    optim = Optim(model, lr = 0.01, weight_decay = 0) 
    epoch_num = 0

    for i in range(0, 1000000):
        input, target, epoch_end_flag = batch.next_batch()

        pred = model(input)
        optim.backward(pred, target)
        
        if epoch_end_flag:
            epoch_num += 1

        if epoch_num % 5 == 0 and epoch_end_flag:
            print("EPOCH Done")
            optim.update_lr(optim.lr/1.2)
            # print(str(optim.lr), model.dropout.p, optim.loss.item())
            # print(str(optim.lr), optim.loss.item())
            print("Epoch " + str(epoch_num) + " : " + str(test(model, batch)))


