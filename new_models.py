class PceFourWayNEW(nn.Module):
    def identity(self, x):
        return x

    def getActivationFn(self, activation):
        if activation == "identity":
            return nn.ReLU()

    def set_dropout(self, dropout):
        self.dropout.p = dropout

    def __init__(self, dim, activation = "identity", dropout = 0.5):
        super(PceFourWayNEW, self).__init__()
        self.dim = dim
        
        self.linear =  nn.Sequential()
        self.linear.add_module("linear", nn.Linear(2*dim, dim))
        self.linear.add_module("activation", self.getActivationFn(activation))
        self.linear_na = nn.Linear(dim, dim)

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
        # print("=====>", r1.shape, output3.shape, A.shape)
        return output3
        # output4 = self.softmax(output3)
        # return output4


