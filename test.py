import pickle
from sklearn.metrics import accuracy_score
from model import *
from train import *

def test(model, batchObj, zero_shot=False, zero_shot_property=None, batch_from="test", no_reverse = False):
    batchObj.reset_batchCount()
    model.eval()

    pred= []
    target = []
    epoch_end_flag = False
    while not epoch_end_flag:
        b, t, epoch_end_flag = batchObj.next_batch(batch_from = batch_from, zero_shot = zero_shot, zero_shot_property=zero_shot_property)
        p = model.forward(b)
        
        if no_reverse:
            pred.extend(list(torch.max(p, 1))[1])
        else:
            b2 = b.chunk(5, 1)
            b2 = torch.cat((b2[1], b2[0], b2[2], b2[3], b2[4]), 1)
            p_tmp = model.forward(b2)

            p2 = torch.FloatTensor(p_tmp.shape[0], 3)
            p2[:, 0] = p[:,0] + p_tmp[:, 2]
            p2[:, 1] = p[:,1] + p_tmp[:,1]
            p2[:, 2] = p[:, 2] + p_tmp[:, 0]

            pred.extend(list(torch.max(p2, 1))[1])

        target.extend(list(t))
        assert(len(pred) == len(target))

    batchObj.reset_batchCount()
    model.train()
    return accuracy_score(pred, target)

def test_emb_similarity(batchObj, batch_from="test"):
    batchObj.reset_batchCount()

    properties = []
    train_data = pickle.load(open("./orig/data/pickle2/verb_physics_train_5.pickle"))
    for i in train_data:
        properties.append(i[2])
    properties = list(set(properties))
    
    for property in properties:
        global pred
        pred= []
        target = []

        b, t, epoch_end_flag = batchObj.next_batch(batch_from = batch_from, zero_shot=True, zero_shot_property=property)
        # print("Test batch size: " + str(len(b)))
        b = b.chunk(5, 1)
        x = b[0]
        y = b[1]
        r1 = b[2]
        r2 = b[3]
        r3 = b[4]

        for i in range(0, len(t)):
            m1 = np.dot(x[i], r1[i])
            m2 = np.dot(x[i], r2[i])
            m3 = np.dot(x[i], r3[i])
            
            if m1 > m2 and m1 > m3:
                m_ind = 0
                m = m1
            elif m2> m1 and m2>m3:
                m_ind = 1
                m = m2
            else:
                m_ind = 2
                m= m3

            n1 = np.dot(y[i], r1[i])
            n2 = np.dot(y[i], r2[i])
            n3 = np.dot(y[i], r3[i])

            if n1 > n2 and n1 > n3:
                n_ind = 0
                n = n1
            elif n2> n1 and n2>n3:
                n_ind = 1
                n = n2
            else:
                n_ind = 2
                n= n3

            if m_ind  == 2-n_ind:
                pred.append(m_ind)
            # elif m > n:
                # pred.append(m_ind)
            else:
                pred.append(0)


            # greater_than = m1 + n3
            # less_than = m3 + n1
            # sim = m2 + n2
            greater_than = m1 - n1
            less_than = m3 - n3
            sim = abs(m2 - n2)
            # sim = 0
            # sim = -100

            # if sim < 0.1:
                # pred.append(0)
            # if sim > greater_than and sim > less_than:
                # pred.append(0)
            # elif greater_than > less_than and greater_than > sim:
                # pred.append(1)
            # else:
                # pred.append(-1)

        print(len([i for i in pred if i == 1]))
        # pred.extend(list(torch.max(p, 1))[1])
        target.extend(list(t))
        assert(len(pred) == len(target))

        batchObj.reset_batchCount()
        # model.train()
        print property + " :" + str(round(accuracy_score(pred, target), 2))

if __name__ == "__main__":
    # batch = Batch(10000, "lstm", "verb_physics", "three_way", 1024)
    batch = Batch(10000, "word2vec", "verb_physics", "three_way", 300)
    batch = Batch(10000, "lstm", "verb_physics", "three_way", 1024)
    print("DEV")
    test_emb_similarity(batch, batch_from = "dev")
    print("\nTEST")
    test_emb_similarity(batch, batch_from = "test")


