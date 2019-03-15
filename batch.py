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

class Batch():
    batchSize = 10
    embeddings = {}
    batchCount = 0

    def __init__(self, batchSize, embeddings_type, dataset, task="3_way", dim = 300):
        self.batchSize = batchSize
        self.embeddings_type = embeddings_type
        self.epoch_num = 0

        vocab_dct = pickle.load(open("./orig/data/pickle2/" + embeddings_type + "/" + embeddings_type + ".6B.vocab.refined.pickle"))
        embeddings_raw = np.load("./orig/data/pickle2/" + 
                                    embeddings_type + "/" + 
                                    embeddings_type + 
                                    ".6B." + str(dim) + 
                                    "d-weights-norm.refined.npy")
        
        self.embeddings_raw = embeddings_raw

        self.dim = len(embeddings_raw[0])
        for word in vocab_dct:
            self.embeddings[word] = embeddings_raw[vocab_dct[word]]

        if dataset == "PCE":
            self.train_data = pickle.load(open("./orig/data/pickle2/train_data.pickle"))
            self.test_data = pickle.load(open("./orig/data/pickle2/test_data.pickle"))
        elif dataset == "verb_physics":
            self.train_data = pickle.load(open("./orig/data/pickle2/verb_physics_train_5.pickle"))
            self.dev_data = pickle.load(open("./orig/data/pickle2/verb_physics_dev_5.pickle"))
            self.test_data = pickle.load(open("./orig/data/pickle2/verb_physics_test_5.pickle"))

        if task == "3_way":
            i = 0
            try:
                all_data = [self.train_data, self.dev_data, self.test_data]
            except:
                all_data = [self.train_data, self.test_data]

            for data in all_data:
                while i < len(data):
                    if data[i][3] == -42:
                        del data[i]
                        continue
                    i += 1

        self.len_train_data = len(self.train_data)
        self.len_test_data = len(self.test_data)

        try:
            self.len_dev_data = len(self.dev_data)
        except:
            pass
        
        self.random_shuffle("train")
        self.random_shuffle("dev")
        self.random_shuffle("test")
        
        self.propadj = eval(open("./orig/data_information/property_adjectives.txt").read())
        
        for i in self.propadj.keys():
            self.propadj[self.propadj[i]] = i


    def set_batchSize(self, batchSize):
        self.batchSize = batchSize

    def reset_batchCount(self):
        self.batchCount = 0

    def random_shuffle(self, which_data = "train"):
        if which_data == "train":
            self.train_data = random.sample(self.train_data, self.len_train_data)
        elif which_data == "dev":
            try:
                self.dev_data = random.sample(self.dev_data, self.len_dev_data)
            except:
                print("EXCEPTION: NO DEV DATA: SKIPPING")
        elif which_data == "test":
            self.test_data = random.sample(self.test_data, self.len_test_data)

    def next_batch(self, batch_from="train", zero_shot=False, zero_shot_property=None):
        #zero-shot property = property NOT to be used while training
        epoch_end_flag = False

        if batch_from == "train":
            data = self.train_data
            len_data = self.len_train_data
        elif batch_from == "dev":
            data = self.dev_data
            len_data = self.len_dev_data
        elif batch_from == "test":
            data = self.test_data
            len_data = self.len_test_data

        s = self.batchCount
        e = min(len_data, self.batchCount + self.batchSize)

        # batch = torch.FloatTensor(self.batchSize, 5 * self.dim)
        # target = torch.LongTensor(self.batchSize)

        batch = []
        target = []

        embeddings = self.embeddings
        for i in range(s, e):
            #ignore word embeddings not present
            try:
                x = embeddings[data[i][0]]
                y = embeddings[data[i][1]]

                r1 = embeddings[data[i][2]]
                r2 = embeddings["similar"]
                r3 = embeddings[self.propadj[data[i][2]]]

                tmp = np.concatenate((x,y))
                tmp = np.concatenate((tmp,r1))
                tmp = np.concatenate((tmp,r2))
                tmp = np.concatenate((tmp,r3))
                # batch[i-s] = torch.FloatTensor(tmp)
                batch.append(tmp)

                if data[i][3] != -42:
                    # target[i-s] = -data[i][3] + 1
                    target.append(-data[i][3]+1)
                else:
                    # target[i-s] = 3
                    target.append(3)
            except:
                pass
        batch = torch.FloatTensor(np.array(batch))
        target = torch.LongTensor(target) 
        self.batchCount += len(batch)

        #### If EPOCH DONE ################
        if self.batchCount >= len_data:
            batch = batch[0: e-s]
            target = target[0: e-s]
            self.batchCount = 0
            self.random_shuffle(which_data = batch_from)
            epoch_end_flag = True

        return batch, target, epoch_end_flag


