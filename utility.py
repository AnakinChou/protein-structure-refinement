import torch
import config
import numpy as np
import matplotlib.pyplot as plt
import os 
import glob

def print_pdb(path_to_original_pdb, coords, path_to_refined_pdb):
    original_pdb = open(path_to_original_pdb, 'r')
    refined_pdb = open(path_to_refined_pdb, 'w')
    i = 0
    for line in original_pdb:
        records = line.split()
        if len(records) < 8:
            continue
        if records[0] != 'ATOM' or records[2] not in config.HEAVY_ATOM:
            continue
        records[5], records[6], records[7] = coords[i][0], coords[i][1], coords[i][2]
        print("{:4}{:>7}  {:<4}{:3}{:>6}{:>12.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}".format(
            records[0], records[1], records[2], records[3], records[4], records[5], records[6],
            records[7], 1.00, 0.00),
            file=refined_pdb)
        i += 1

def plot_loss_fig(filename):
    '''filename: loss.txt'''
    time_stamp = os.path.basename(filename).rsplit('_',maxsplit=1)[0]
    savePath = os.path.join(os.path.dirname(filename),time_stamp + '_loss.png')
    y = np.loadtxt(filename)
    x = np.arange(0,y.shape[0])
    plt.scatter(x,y)
    plt.savefig(savePath)
    # plt.show()
    plt.close()
    

def batch_plot_loss_fig(dataPath):
    for target in os.listdir(dataPath):
        target = os.path.join(dataPath,target)
        if not os.path.isdir(target):
            continue      
        for file in os.listdir(target):
            if file.endswith('loss.txt'):
                plot_loss_fig(os.path.join(target,file))


def results_statist(dataPath):
    tmscore = '/home/anakin/Software/TMscore'
    init = True
    for target in os.listdir(dataPath):
        print(target)
        native_addr = os.path.join(dataPath, target, target+'.pdb')
        init_addr = os.path.join(dataPath, target, '1.pdb')
        target = os.path.join(dataPath,target)
        if not os.path.isdir(target):
            continue 
        refined_models = glob.glob(target+'/refined*')
        refined_models.insert(0,init_addr)
        tmfile = os.path.join(target,'tmscore.txt')
        while len(refined_models)<3:
            refined_models.append(refined_models[-1])
        score = {'gtd_ts': [], 'tmscore': [], 'rmsd': []}
        for model in refined_models:

            command = ' '.join([tmscore, native_addr, model, '>', tmfile])

            os.system(command)
            with open(tmfile, 'r') as f:
                for line in f.readlines():
                    splits = line.split()
                    if (splits == []):
                        continue
                    if (splits[0] == 'GDT-TS-score='):
                        score['gtd_ts'].append(float(splits[1]))
                    if (splits[0] == 'TM-score'):
                        score['tmscore'].append(float(splits[2]))
                    if (splits[0] == 'RMSD'):
                        score['rmsd'].append(float(splits[5]))
            # os.remove(tmfile)
        with open(os.path.join(dataPath,'statistics.txt'),'a') as g:
            if init:
                print('target    gdt_ts/tmscore/rmsd    gdt_ts/tmscore/rmsd    gdt_ts/tmscore/rmsd',file=g)
                init=False
            print('{:12}{:5}/{:5}/{:<9}{:5}/{:5}/{:<9}{:5}/{:5}/{:<5}'
                .format(os.path.basename(target),score['gtd_ts'][0],score['tmscore'][0],score['rmsd'][0],
                        score['gtd_ts'][1],score['tmscore'][1],score['rmsd'][1],
                        score['gtd_ts'][2],score['tmscore'][2],score['rmsd'][2]),file=g)

                        
if __name__=='__main__':
    # results_statist('./data')
    plot_loss_fig('/home/anakin/refinement/data/R1042v1/0530_2046_loss.txt')
