from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import torch
from torch.nn import functional
import dgl
import os
import sys
from NeRF_torch import cal_coords_forO
import warnings
import numpy as np
from random import sample
import glob
import tqdm
import shutil
from multiprocessing import Pool

warnings.filterwarnings("ignore")

# helix: 1 0 0
# strand 0 1 0
# coil   0 0 1
ss_to_code = {
    'G': 0,
    'H': 0,
    'I': 0,
    'T': 1,
    'E': 1,
    'B': 1,
    'S': 1,
    '-': 2
}
# 1LAA to number conversion
aa = 'ACDEFGHIKLMNPQRSTVWY-'
aa_to_code = dict([(aa[i], i) for i in range(len(aa))])
TINY = 1e-8


# N CA C O
# 0 -  1 2
class DataStudio():
    def __init__(self, src_file):
        if not os.path.exists(src_file):
            print("fatal error: %s does not exist" % src_file, file=sys.stderr)
            exit(1)
        self.file = src_file
        self.coords = cal_coords_forO(src_file)
        self.bins = [1, 2, 3, 4, 5, 6, 11, 16, 21]
        self.neighbor_threshold = 10
        self.u = None
        self.v = None
        self.nft = None  # Node feature: aa(21) rPosition(1) dihedral(2) ss(3) RSA(1)
        self.eft = None  # edge feature
        self.gft = None  # graph feature
        self.dssp = None

    def get_node_feature(self):
        self.exec_dssp()
        length = len(self.dssp)
        assert length != 0
        aa_code = functional.one_hot(torch.tensor(aa_to_code[self.dssp[0][1]]), 21).to(torch.float32)
        rPosition = torch.tensor([0 / length], dtype=torch.float32)
        ss_code = functional.one_hot(torch.tensor(ss_to_code[self.dssp[0][2]]), 3).to(torch.float32)
        dihedral_rsa = torch.tensor([self.dssp[0][4] / 180 * np.pi, self.dssp[0][5] / 180 * np.pi, self.dssp[0][3]],
                                    dtype=torch.float32)
        # atom_code = torch.cat([
        #     torch.tensor([1,0,0],dtype=torch.float32),
        #     self.coords[0]-self.coords[1],
        #     torch.tensor([0,1,0],dtype=torch.float32),
        #     self.coords[2]-self.coords[1],
        #     torch.tensor([0,0,1],dtype=torch.float32),
        #     self.coords[3]-self.coords[1],
        # ])
        atom_code = self.get_atom_code(0)
        # print(atom_code)
        f = torch.cat([aa_code, rPosition, ss_code, dihedral_rsa, atom_code]).to(torch.float32)
        self.nft = f.view(1, -1)
        for i in range(1, len(self.dssp)):
            aa_code = functional.one_hot(torch.tensor(aa_to_code[self.dssp[i][1]]), 21).to(torch.float32)
            rPosition = torch.tensor([i / length], dtype=torch.float32)
            ss_code = functional.one_hot(torch.tensor(ss_to_code[self.dssp[i][2]]), 3).to(torch.float32)
            dihedral_rsa = torch.tensor([self.dssp[i][4] / 180 * np.pi, self.dssp[i][5] / 180 * np.pi, self.dssp[i][3]],
                                        dtype=torch.float32)
            # atom_code = torch.cat([
            #     torch.tensor([1,0,0],dtype=torch.float32),
            #     self.coords[0+i*4]-self.coords[1+i*4],
            #     torch.tensor([0,1,0],dtype=torch.float32),
            #     self.coords[2+i*4]-self.coords[1+i*4],
            #     torch.tensor([0,0,1],dtype=torch.float32),
            #     self.coords[3+i*4]-self.coords[1+i*4],
            # ])
            atom_code = self.get_atom_code(i)
            f = torch.cat([aa_code, rPosition, ss_code, dihedral_rsa, atom_code]).to(torch.float32).view(1, -1)
            self.nft = torch.cat([self.nft, f])
        # print(self.nft)

    # local frame originates at CA
    def get_atom_code(self, k):
        N, CA, C, O = self.coords[0 + k * 4], self.coords[1 + k * 4], self.coords[2 + k * 4], self.coords[3 + k * 4]
        CA_N = (N - CA) / torch.norm(N - CA)
        CA_C = (C - CA) / torch.norm(C - CA)
        nz = torch.cross(CA_N, CA_C) / torch.norm(torch.cross(CA_N, CA_C))
        ny = torch.cross(nz, CA_N) / torch.norm(torch.cross(nz, CA_N))
        mat = torch.inverse(torch.stack([CA_N, ny, nz], dim=1))
        return torch.cat([
            torch.tensor([1, 0, 0], dtype=torch.float32),
            mat @ (self.coords[0 + k * 4] - self.coords[1 + k * 4]),
            torch.tensor([0, 1, 0], dtype=torch.float32),
            mat @ (self.coords[2 + k * 4] - self.coords[1 + k * 4]),
            torch.tensor([0, 0, 1], dtype=torch.float32),
            mat @ (self.coords[3 + k * 4] - self.coords[1 + k * 4]),
        ])

    def exec_dssp(self):
        p = PDBParser()
        structure = p.get_structure("1MOT", self.file)  # "1MOT" doesn't matter
        model = structure[0]
        dssp = DSSP(model, self.file, dssp='mkdssp')
        keys = list(dssp.keys())
        self.dssp = [dssp[key][:6] for key in keys]

    def get_edge_feature(self):
        if self.u is None or self.v is None:
            self.make_graph()
        len_edges = len(self.u)
        for i in range(len_edges):
            dis = torch.norm(self.coords[1 + 4 * self.u[i]] - self.coords[1 + 4 * self.v[i]]).to(torch.float32)
            orientation = self.get_orientation(self.u[i], self.v[i])
            e = torch.cat([dis.unsqueeze(0), orientation]).view(1, -1).float()
            if self.eft is None:
                self.eft = e
            else:
                self.eft = torch.cat([self.eft, e])

    def get_orientation(self, i, j):
        N, CA, C, CA_dst = self.coords[0 + i * 4], self.coords[1 + i * 4], self.coords[2 + i * 4], self.coords[
            1 + j * 4]
        CA_N = (N - CA) / torch.norm(N - CA)
        CA_C = (C - CA) / torch.norm(C - CA)
        nz = torch.cross(CA_N, CA_C) / torch.norm(torch.cross(CA_N, CA_C))
        ny = torch.cross(nz, CA_N) / torch.norm(torch.cross(nz, CA_N))
        mat = torch.inverse(torch.stack([CA_N, ny, nz], dim=1))
        new_dst = mat @ (CA_dst - CA)
        theta = torch.acos(new_dst[2] / torch.norm(new_dst))
        phi = torch.atan(new_dst[1] / (new_dst[0] + TINY))
        sequential_separation = np.digitize(np.abs(i - j), bins=self.bins) - 1
        sep_code = functional.one_hot(torch.tensor(sequential_separation), len(self.bins))
        return torch.cat([theta.unsqueeze(0), phi.unsqueeze(0), sep_code])

    # 10 A
    def make_graph(self):
        edges = []
        length = len(self.coords) // 4
        for i in range(length):
            for j in range(length):
                if i == j:
                    continue
                if torch.norm(self.coords[1 + i * 4] - self.coords[1 + j * 4]).item() <= self.neighbor_threshold:
                    edges.append((i, j))
        self.u, self.v = zip(*edges)

    def make_label(self, nativeFile):
        if self.u is None or self.v is None:
            self.make_graph()
        len_edges = len(self.u)
        native_coords = cal_coords_forO(nativeFile)
        label = []
        for i in range(len_edges):
            dis = torch.norm(native_coords[1 + 4 * self.u[i]] - native_coords[1 + 4 * self.v[i]]).to(torch.float32)
            label.append(dis.item())
        return torch.tensor(label, dtype=torch.float32)

    def get_graph_feature(self):
        self.get_node_feature()
        self.get_edge_feature()


def save_graphs(src, src_native, dst):
    x = DataStudio(src)
    x.get_graph_feature()
    label = x.make_label(src_native)
    if torch.any(torch.isnan(x.nft)) or torch.any(torch.isnan(x.eft)) or torch.any(torch.isnan(label)):
        print("Fatal error: {} hold Nan data".format(src))
        return
    g = dgl.graph((torch.tensor(x.u), torch.tensor(x.v)))
    g.ndata['h'] = x.nft
    g.edata['h'] = x.eft
    dgl.save_graphs(dst, g, {'label': label})


def save_by_folder(folder, pdbs, traindata):
    files = glob.glob(os.path.join(pdbs, folder, '*.pdb'))
    native_pdb = os.path.join(pdbs, folder, 'native.pdb')
    # print(native_pdb)
    newFoler = os.path.join(traindata, folder)
    if os.path.exists(newFoler):
        shutil.rmtree(newFoler)
    os.mkdir(newFoler)
    if native_pdb not in files:
        print('{} has no native file'.format(folder))
        return
    for file in tqdm.tqdm(files):
        basename = os.path.basename(file)
        dst = os.path.join(newFoler, basename[:-4] + '.bin')
        # print(dst)
        save_graphs(file, native_pdb, dst)


if __name__ == "__main__":
    pdbs = '/data/zhouchengpeng/dataset/pdbs'
    traindata = 'traindata'
    folderList = sample([fold for fold in os.listdir(pdbs) if os.path.isdir(fold)], 900)
    num_p = 40
    p = Pool(num_p)
    for folder in tqdm.tqdm(folderList):
        newFoler = os.path.join(traindata, folder)
        # if os.path.exists(newFoler):
        #     continue
        p.apply_async(save_by_folder, args=(folder, pdbs, traindata))
    p.close()
    p.join()
    print("Success")

    # print(x.nft.shape)
