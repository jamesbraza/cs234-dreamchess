#!/usr/bin/env python

from alpha_net import ChessNet, train
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp


if __name__=="__main__":
    net_to_train="iter2_multi.tar"; save_as="iter1_2_12_multi.tar"
    # gather data
    data_path = "./datasets/iter2/"
    datasets = []
    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))

    datasets = np.array(datasets)
    
    # initiate Net
    mp.set_start_method("spawn",force=True)
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    net.train()
    print("hi")
    current_net_filename = os.path.join("./model_data/",  net_to_train)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])
    
    processes = []
    for i in range(20):
        p = mp.Process(target=train,args=(net,datasets,0,10,i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    # save results
    t = os.path.join("./model_data/",save_as)
    torch.save({'state_dict': net.state_dict()}, t)
    print(t)
