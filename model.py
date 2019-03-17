import torch
import torch.nn as nn
import torch.nn.functional as F

class PceFourWay(nn.Module):
    def identity(self, x):
        return x

    def getActivationFn(self, activation):
        if activation == "identity":
            return nn.ReLU()

    def set_dropout(self, dropout):
        self.dropout.p = dropout

    def __init__(self, dim, activation = "identity", dropout = 0.5, ignore_similar_emb = False):
        super(PceFourWay, self).__init__()
        self.dim = dim
        
        self.linear =  nn.Sequential()
        self.linear.add_module("linear", nn.Linear(2*dim, dim))
        self.linear.add_module("activation", self.getActivationFn(activation))
        self.linear_na = nn.Linear(dim, dim)

        if ignore_similar_emb:
            self.similar_embedding_nn = nn.Sequential()
            self.similar_embedding_nn.add_module("linear", nn.Linear(2*dim, dim))
            self.similar_embedding_nn.add_module("activation", nn.ReLU())

        self.ignore_similar_emb = ignore_similar_emb
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_data):
        batchSize = input_data.shape[0]
        input = self.dropout(input_data)
        input = input.chunk(5, 1)

        x = input[0]
        y = input[1]
        r1 = input[2]
        r2 = input[3]
        r3 = input[4]
        
        if self.ignore_similar_emb:
            r13 = torch.cat((r1, r2), 1)
            r2 = self.similar_embedding_nn(r13)
            # r2 = (r1 + r3) / 2

        r1 = r1.unsqueeze(1)
        r2 = r2.unsqueeze(1)
        r3 = r3.unsqueeze(1)

        R = torch.cat((r1, r2), 1)
        R = torch.cat((R, r3), 1)

        hx = self.linear_na(x).unsqueeze(1)
        hy = self.linear_na(y).unsqueeze(1)
        hx = hx.view(hx.shape[0], self.dim, 1)
        hy = hy.view(hy.shape[0], self.dim, 1)

        Ax = torch.bmm(r1+r3, hx)
        Ay = torch.bmm(r1+r3, hy)
        A = Ax +  Ay
        A = A.squeeze(2)

        xy = torch.cat((x, y), 1)
        output1 = self.linear(xy)
        output2 = output1.unsqueeze(2)
        output3 = torch.bmm(R, output2).squeeze(2)

        output3 = torch.cat((output3, A), 1)
        return output3

class PceThreeWay(nn.Module):
    def identity(self, x):
        return x

    def getActivationFn(self, activation):
        if activation == "identity":
            return nn.ReLU()

    def set_dropout(self, dropout):
        self.dropout.p = dropout

    def __init__(self, dim, activation = "identity", dropout = 0.5):
        super(PceThreeWay, self).__init__()
        self.dim = dim
        
        self.linear =  nn.Sequential()
        self.linear.add_module("linear", nn.Linear(2*dim, dim))
        self.linear.add_module("activation", self.getActivationFn(activation))
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_data):
        batchSize = input_data.shape[0]
        input = self.dropout(input_data)
        input = input.chunk(5, 1)

        x = input[0]
        y = input[1]

        r1 = input[2]
        r2 = input[3]
        r3 = input[4]

        r1 = r1.unsqueeze(1)
        r2 = r2.unsqueeze(1)
        r3 = r3.unsqueeze(1)
        
        R = torch.cat((r1, r2), 1)
        R = torch.cat((R, r3), 1)

        xy = torch.cat((x, y), 1)
        output1 = self.linear(xy)
        output2 = output1.unsqueeze(2)
        output3 = torch.bmm(R, output2).squeeze(2)
        return output3

class Optim:
    def __init__(self, model, criterion = nn.CrossEntropyLoss(), lr = 0.001, weight_decay=0):
        self.model = model
        self.criterion = criterion;
        self.optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        self.lr = lr

        self.weight_decay = weight_decay

    def update_lr(self, lr, weight_decay = None):
        if weight_decay == None:
            weight_decay = self.weight_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        self.lr = lr

    def backward(self, output, y):
        loss = self.criterion(output, y)
        self.loss = loss
        # print(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class PceOnePole(nn.Module):
    def identity(self, x):
        return x

    def getActivationFn(self, activation):
        if activation == "identity":
            return nn.ReLU()

    def set_dropout(self, dropout):
        self.dropout.p = dropout

    def __init__(self, dim, activation = "identity", dropout = 0.5):
        super(PceOnePole, self).__init__()
        self.dim = dim
        
        self.linear =  nn.Sequential()
        self.linear.add_module("linear", nn.Linear(2*dim, dim))
        self.linear.add_module("activation", self.getActivationFn(activation))
        self.linear_similar = nn.Linear(dim, dim)
        self.linear_antonym = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_data):
        batchSize = input_data.shape[0]
        input = self.dropout(input_data)
        input = input.chunk(5, 1)

        x = input[0]
        y = input[1]

        r1 = input[2]
        r2 = input[3]
        r3 = input[4]

        r2 = self.linear_similar(r1)
        r3 = self.linear_antonym(r1)

        r1 = r1.unsqueeze(1)
        r2 = r2.unsqueeze(1)
        r3 = r3.unsqueeze(1)
        
        R = torch.cat((r1, r2), 1)
        R = torch.cat((R, r3), 1)

        xy = torch.cat((x, y), 1)
        output1 = self.linear(xy)
        output2 = output1.unsqueeze(2)
        output3 = torch.bmm(R, output2).squeeze(2)


        o1 = torch.bmm(r1, output2).squeeze(2)
        o2 = torch.bmm(r2, output2).squeeze(2)
        o3 = torch.bmm(r3, output2).squeeze(2)

        # print(output3.shape)
        output3 = torch.stack((o1, o2, o3), 1).squeeze(2)
        # print(output3.shape)
        return output3
