import numpy as np



def acc_cmf(y_coarse, y_middle, y_fine,y_testfine1 , dataset):
    not_nested = []
    error_coarse = []
    error_middle = []
    error_fine = []

    perm_mnist = [3,5,8,6,0,4,7,9,2,1]
    perm_svhn = [3,5,8,6,0,4,7,9,2,1]
    perm_cifar10 = [0,1,8,9,2,6,3,5,4,7]
    perm_fmnist = [0,2,6,3,4,5,7,9,1,8]
    perm = [0,1,2,3,4,5,6,7,8,9]
    if dataset == 'cifar10':
        perm = perm_cifar10
    elif dataset == 'mnist':
        perm = perm_mnist
    elif dataset == 'fashion_mnist':
        perm = perm_fmnist
    elif dataset == 'SVHN':
        perm = perm_svhn


    n = y_coarse.shape[0]
    for i in range(n):
        y1 = y_coarse[i,:]
        y2 = y_middle[i,:]
        y3 = y_fine[i,:]
        if np.argmax(y3)!=y_testfine1[i]:
            not_correct.add(i)
            error_fine.append(i)
        if dataset == 'cifar10':
            if np.argmax(y2)==0 and not(y_testfine1[i] in perm[0:2]):
                not_correct.add(i)
                error_middle.append(i)
            elif np.argmax(y2)==1 and not(y_testfine1[i] in perm[2:4]):
                not_correct.add(i)
                error_middle.append(i)
            elif np.argmax(y2)==2 and not(y_testfine1[i] in perm[4:6]):
                not_correct.add(i)
                error_middle.append(i)
            elif np.argmax(y2)==3 and not(y_testfine1[i] in perm[6:8]):
                not_correct.add(i)
                error_middle.append(i)
            elif np.argmax(y2)==4 and not(y_testfine1[i] in perm[8:10]):
                not_correct.add(i)
                error_middle.append(i)
            if np.argmax(y1)==0 and not(y_testfine1[i] in perm[0:4]):
                not_correct.add(i)
                error_coarse.append(i)
            elif np.argmax(y1)==1 and not(y_testfine1[i] in perm[4:]):
                not_correct.add(i)
                error_coarse.append(i)
        else :
            if np.argmax(y2)==0 and not(y_testfine1[i] in perm[0:3]):
                not_correct.add(i)
                error_middle.append(i)
            elif np.argmax(y2)==1 and not(y_testfine1[i] in perm[3:5]):
                not_correct.add(i)
                error_middle.append(i)
            elif np.argmax(y2)==2 and not(y_testfine1[i] in perm[5:8]):
                not_correct.add(i)
                error_middle.append(i)
            elif np.argmax(y2)==3 and not(y_testfine1[i] in perm[8:]):
                not_correct.add(i)
                error_middle.append(i)
            if np.argmax(y1)==0 and not(y_testfine1[i] in perm[0:5]):
                not_correct.add(i)
                error_coarse.append(i)
            elif np.argmax(y1)==1 and not(y_testfine1[i] in perm[5:]):
                not_correct.add(i)
                error_coarse.append(i)
    acc_c = (n - len(error_coarse))/float(n)
    acc_m = (n - len(error_middle))/float(n)
    acc_f = (n - len(error_fine))/float(n)
    return(acc_c, acc_m, acc_f)

def acc_cmf_single(y_pred,y_test_fine,dataset):
    perm_mnist = [3,5,8,6,0,4,7,9,2,1]
    perm_svhn = [3,5,8,6,0,4,7,9,2,1]
    perm_cifar10 = [0,1,8,9,2,6,3,5,4,7]
    perm_fmnist = [0,2,6,3,4,5,7,9,1,8]
    perm = [0,1,2,3,4,5,6,7,8,9]
    if dataset == 'cifar10':
        perm = perm_cifar10
    elif dataset == 'mnist':
        perm = perm_mnist
    elif dataset == 'fashion_mnist':
        perm = perm_fmnist
    elif dataset == 'SVHN':
        perm = perm_svhn
    n = y_pred.shape[0]
    c = 0
    m = 0
    f = 0
    # fn = 0.
    # cc = 0.
    # mm = 0.
    for i in range(n):
        pred = np.argmax(y_pred[i,:])
        true = np.argmax(y_test_fine[i,:])
        if pred == true :
            f+=1
        # else :
        #     fn+=1
        #     if (pred in [3,5,8]) and (true in [3,5,8]):
        #         mm+=1
        #     if (pred in [0,6]) and (true in [0,6]):
        #         mm+=1
        #     if (pred in [4,7,9]) and (true in [4,7,9]):
        #         mm+=1
        #     if (pred in [1,2]) and (true in [1,2]):
        #         mm+=1
        #     if (pred in [0,3,5,8,6]) and (true in [0,3,5,8,6]):
        #         cc+=1
        #     if (pred in [1,2,4,7,9]) and (true in [1,2, 4, 7, 9]):
        #         cc+=1
        if dataset == 'cifar10':
            pass
        else :
            if (pred in perm[0:3]) and (true in perm[0:3]):
                m+=1
            if (pred in perm[3:5]) and (true in perm[3:5]):
                m+=1
            if (pred in perm[5:8]) and (true in perm[5:8]):
                m+=1
            if (pred in perm[8:]) and (true in perm[8:]):
                m+=1
            if (pred in perm[0:5]) and (true in perm[0:5]):
                c+=1
            if (pred in perm[5:]) and (true in perm[5:]):
                c+=1
    acc_c = c/n
    acc_m = m/n
    acc_f = f/n
    return(acc_c,acc_m,acc_f)
def conf_cmf_single(y_pred,dataset):
    perm_mnist = [3,5,8,6,0,4,7,9,2,1]
    perm_svhn = [3,5,8,6,0,4,7,9,2,1]
    perm_cifar10 = [0,1,8,9,2,6,3,5,4,7]
    perm_fmnist = [0,2,6,3,4,5,7,9,1,8]
    perm = [0,1,2,3,4,5,6,7,8,9]
    if dataset == 'cifar10':
        perm = perm_cifar10
    elif dataset == 'mnist':
        perm = perm_mnist
    elif dataset == 'fashion_mnist':
        perm = perm_fmnist
    elif dataset == 'SVHN':
        perm = perm_svhn
    n = y_pred.shape[0]
    conf_c, conf_m, conf_f = 0
    conf_f = np.mean(np.max(y_pred,axis = 1))

    if dataset =='cifar10':
        fine2middle = np.zeros(n,5)
        fine2coarse = np.zeros(n,2)
        for i in range(n):
            fine2middle[i,0] = np.sum(y_pred[i,perm[0:2]])
            fine2middle[i,1] = np.sum(y_pred[i,perm[2:4]])
            fine2middle[i,2] = np.sum(y_pred[i,perm[4:6]])
            fine2middle[i,3] = np.sum(y_pred[i,perm[6:8]])
            fine2middle[i,4] = np.sum(y_pred[i,perm[8:]])
            fine2coarse[i:0] = np.sum(y_pred[i,perm[0:4]])
            fine2coarse[i:1] = np.sum(y_pred[i,perm[4:]])
    else :
        fine2middle = np.zeros(n,4)
        fine2coarse = np.zeros(n,2)
        for i in range(n):
            fine2middle[i,0] = np.sum(y_pred[i,perm[0:3]])
            fine2middle[i,1] = np.sum(y_pred[i,perm[3:5]])
            fine2middle[i,2] = np.sum(y_pred[i,perm[5:8]])
            fine2middle[i,3] = np.sum(y_pred[i,perm[8:]])
            fine2coarse[i:0] = np.sum(y_pred[i,perm[0:5]])
            fine2coarse[i:1] = np.sum(y_pred[i,perm[5:]])
    conf_m = np.mean(np.max(fine2middle,axis = 1))
    conf_c = np.mean(np.max(fine2coarse,axis = 1))
    return(conf_c,conf_m, conf_f)
