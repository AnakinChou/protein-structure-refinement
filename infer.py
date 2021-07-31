import torch
import torch.nn as nn
import os
import dgl
from model import GraphTransformerNet
from NeRF_LBFGS import cal_distogram, cal_coords
import numpy as np
import tqdm


def pred_distogram(initial_file, dic):
    distogram = cal_distogram(cal_coords(initial_file)).detach()
    # print(distogram)
    u, v, pred = dic['u'], dic['v'], dic['pred']
    for i in range(len(u)):
        distogram[u[i], v[i]] = pred[i]
    # print(distogram)
    return distogram


def get_distogram(graph_folder, pdb_folfer, target, model):
    g, _ = dgl.load_graphs(os.path.join(graph_folder, target, '1.bin'))
    u, v = g[0].edges()
    pred = model(g[0], g[0].ndata['h'], g[0].edata['h'])
    pred_dic = {'u': u, 'v': v, 'pred': pred}
    distogram = pred_distogram(os.path.join(pdb_folfer, target, '1.pdb'), pred_dic).detach().numpy()
    save_path = os.path.join(pdb_folfer, target, 'pre_dis.npy')
    np.save(save_path, distogram)


if __name__ == '__main__':
    graph_folder = 'casp14'
    pdb_folder = 'data'
    model_path = 'models/model_64_1_2.pth'
    model = GraphTransformerNet(46, 12, 64, 1, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for target in tqdm.tqdm(os.listdir(graph_folder)):
        if not os.path.isdir(os.path.join(graph_folder, target)):
            continue
        get_distogram(graph_folder, pdb_folder, target, model)
