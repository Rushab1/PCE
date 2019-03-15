import pickle
from sklearn.metrics import accuracy_score
from model import *
from train import *

def test(model, batchObj):
    batchObj.reset_batchCount()
    model.eval()

    pred= []
    target = []
    epoch_end_flag = False
    while not epoch_end_flag:
        b, t, epoch_end_flag = batchObj.next_batch(batch_from = "test")
        b2 = b.chunk(5, 1)
        b2 = torch.cat((b2[1], b2[0], b2[4], b2[3], b2[2]), 1)
        
        global p, p2

        p = model.forward(b)
        p2 = model.forward(b2)
        
        try:
            pred.extend(list(torch.max(p+p2, 1))[1])
        except Exception as e:
            return "alsdjaldjslk"
        target.extend(list(t))
        assert(len(pred) == len(target))

    batchObj.reset_batchCount()
    model.train()
    return accuracy_score(pred, target)

