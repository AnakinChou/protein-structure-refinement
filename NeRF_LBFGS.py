import numpy as np
import time
import torch
from utility import print_pdb
import os
import sys
import datetime

MAIN_CHAIN = ['N', 'CA', 'C']
HEAVY_ATOM = ['N', 'CA', 'C', 'O']
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def findAngle(v1, v2, reference):
    """calculate angle"""
    radian = torch.acos(torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2)))
    # deg = radian * 180 / np.pi
    deg = radian
    v3 = torch.cross(v1, v2)
    if torch.dot(v3, reference) < 0:
        deg = -deg
    return deg


def cal_coords(filename):
    """input file"""
    file = open(filename, 'r')
    cartesian_coordinate = []

    for line in file:
        content = line.split()
        if len(content) < 3:
            continue
        if content[0] == 'ATOM' and content[2] in MAIN_CHAIN:
            cartesian_coordinate.append([float(content[5]), float(content[6]), float(content[7])])

    cartesian_coordinate = torch.tensor(cartesian_coordinate, dtype=torch.float64, device=device)
    # print(cartesian_coordinate.shape)
    return cartesian_coordinate


def cal_coords_forO(filename):
    """input file"""
    file = open(filename, 'r')
    cartesian_coordinate = []

    for line in file:
        content = line.split()
        if len(content) < 3:
            continue
        if content[0] == 'ATOM' and content[2] in HEAVY_ATOM:
            cartesian_coordinate.append([float(content[5]), float(content[6]), float(content[7])])

    cartesian_coordinate = torch.tensor(cartesian_coordinate, dtype=torch.float64, device=device)
    # print(cartesian_coordinate.shape)
    return cartesian_coordinate


def convert_to_internal(cartesian_coordinate):
    """convert cartesian to internal coordinates"""
    dihedral = []
    bond_angle = []
    bond_length = []
    for i in range(1, cartesian_coordinate.shape[0] - 2):
        N_CA = cartesian_coordinate[i] - cartesian_coordinate[i - 1]
        CA_C = cartesian_coordinate[i + 1] - cartesian_coordinate[i]
        C_N = cartesian_coordinate[i + 2] - cartesian_coordinate[i + 1]
        n_NCAC = torch.cross(N_CA, CA_C)
        n_CACN = torch.cross(CA_C, C_N)
        bond_angle.append(findAngle(-CA_C, C_N, torch.cross(-CA_C, C_N)))
        dihedral.append(findAngle(n_NCAC, n_CACN, CA_C))
        bond_length.append(torch.norm(C_N))
    # print(dihedral.__len__())
    # print(bond_angle)
    # print(bond_length)
    return torch.tensor(dihedral, dtype=torch.float64, requires_grad=True, device=device), \
           torch.tensor(bond_angle, dtype=torch.float64, device=device), \
           torch.tensor(bond_length, dtype=torch.float64, device=device)


def convert_to_internal_forO(cartesian_coordinate):
    """convert cartesian to internal coordinates"""
    dihedral = []
    bond_angle = []
    bond_length = []
    for i in range(1, cartesian_coordinate.shape[0], 4):
        N_CA = cartesian_coordinate[i] - cartesian_coordinate[i - 1]
        CA_C = cartesian_coordinate[i + 1] - cartesian_coordinate[i]
        C_N = cartesian_coordinate[i + 2] - cartesian_coordinate[i + 1]
        n_NCAC = torch.cross(N_CA, CA_C)
        n_CACN = torch.cross(CA_C, C_N)
        bond_angle.append(findAngle(-CA_C, C_N, torch.cross(-CA_C, C_N)))
        dihedral.append(findAngle(n_NCAC, n_CACN, CA_C))
        bond_length.append(torch.norm(C_N))
    # print(len(dihedral))
    # print(len(bond_angle))
    # print(len(bond_length))
    return torch.tensor(dihedral, dtype=torch.float64, device=device), \
           torch.tensor(bond_angle, dtype=torch.float64, device=device), \
           torch.tensor(bond_length, dtype=torch.float64, device=device)


def internal_to_coords(pred_coords, dihedral, bond_angle, bond_length):
    """convert internal to cartesian coordinates using NeRF"""
    for i in range(dihedral.shape[0]):
        A, B, C = pred_coords[-3], pred_coords[-2], pred_coords[-1]
        bc = (C - B) / torch.norm(C - B)
        AB = B - A
        n = torch.cross(AB, bc) / torch.norm(torch.cross(AB, bc))
        M = torch.stack([bc, torch.cross(n, bc), n], dim=1)
        R, theta, phi = bond_length[i], bond_angle[i], dihedral[i]
        D2 = torch.stack(
            [-R * torch.cos(theta), R * torch.cos(phi) * torch.sin(theta), R * torch.sin(phi) * torch.sin(theta)])
        D = (torch.matmul(M, D2)).squeeze() + C
        pred_coords = torch.cat([pred_coords, D.reshape(1, 3)])
    return pred_coords


def internal_to_coords_withO(pred_coords, dihedral, bond_angle, bond_length, dihedral_O, bond_angle_O, bond_length_O):
    """convert internal to cartesian coordinates using NeRF"""
    index_O = 0
    pred_coords_withO = pred_coords

    for i in range(dihedral.shape[0]):
        A, B, C = pred_coords[-3], pred_coords[-2], pred_coords[-1]
        bc = (C - B) / torch.norm(C - B)
        AB = B - A
        n = torch.cross(AB, bc) / torch.norm(torch.cross(AB, bc))
        M = torch.stack([bc, torch.cross(n, bc), n], dim=1)
        # add atom O
        if i % 3 == 0:
            R, theta, phi = bond_length_O[index_O], bond_angle_O[index_O], dihedral_O[index_O]
            D2 = torch.stack(
                [-R * torch.cos(theta), R * torch.cos(phi) * torch.sin(theta), R * torch.sin(phi) * torch.sin(theta)])
            D = (torch.matmul(M, D2)).squeeze() + C
            pred_coords_withO = torch.cat([pred_coords_withO, D.reshape(1, 3)])
            index_O += 1
        R, theta, phi = bond_length[i], bond_angle[i], dihedral[i]
        D2 = torch.stack(
            [-R * torch.cos(theta), R * torch.cos(phi) * torch.sin(theta), R * torch.sin(phi) * torch.sin(theta)])
        D = (torch.matmul(M, D2)).squeeze() + C
        pred_coords = torch.cat([pred_coords, D.reshape(1, 3)])
        pred_coords_withO = torch.cat([pred_coords_withO, D.reshape(1, 3)])
    # print(index_O)
    # add the last atom O
    A, B, C = pred_coords[-3], pred_coords[-2], pred_coords[-1]
    bc = (C - B) / torch.norm(C - B)
    AB = B - A
    n = torch.cross(AB, bc) / torch.norm(torch.cross(AB, bc))
    M = torch.stack([bc, torch.cross(n, bc), n], dim=1)
    R, theta, phi = bond_length_O[index_O], bond_angle_O[index_O], dihedral_O[index_O]
    D2 = torch.stack(
        [-R * torch.cos(theta), R * torch.cos(phi) * torch.sin(theta), R * torch.sin(phi) * torch.sin(theta)])
    D = (torch.matmul(M, D2)).squeeze() + C
    # pred_coords = np.concatenate([pred_coords, D.reshape(1, 3)])
    pred_coords_withO = torch.cat([pred_coords_withO, D.reshape(1, 3)])
    return pred_coords_withO


def cal_distogram(coords):
    ca_native = coords[1::3]
    x = torch.norm(ca_native - ca_native[0], dim=1).view(1, -1)
    for i in range(1, ca_native.shape[0]):
        dist = torch.norm(ca_native - ca_native[i], dim=1).view(1, -1)
        x = torch.cat([x, dist])
    return x


def quadratic_potential(native_distogram, pred_coords):
    pred_distogram = cal_distogram(pred_coords)
    potential = torch.pow(pred_distogram - native_distogram, 2).sum()
    return potential


def refinement(dirname, learning_rate, iteration):
    filename = os.path.join(dirname, '1.pdb')
    native_file = os.path.join(dirname, os.path.split(dirname)[1] + '.pdb')

    log_file = os.path.join(dirname, 'log.txt')
    log = open(log_file, 'w')
    cartesian_coords = cal_coords(filename)
    cartesian_coords_forO = cal_coords_forO(filename)
    dihedral, bond_angle, bond_length = convert_to_internal(cartesian_coords)
    dihedral_O, bond_angle_O, bond_length_O = convert_to_internal_forO(cartesian_coords_forO)
    # First Three Atoms
    first_three_atoms = torch.stack([cartesian_coords[0], cartesian_coords[1], cartesian_coords[2]])
    # NeRF
    pred_coords = internal_to_coords(first_three_atoms, dihedral, bond_angle, bond_length)

    native_coords = cal_coords(native_file)
    # native_distogram = cal_distogram(native_coords)
    native_distogram = torch.from_numpy(np.load(os.path.join(dirname, 'pre_dis.npy'))).to(torch.float64)
    optimizer = torch.optim.LBFGS({dihedral}, lr=learning_rate)
    # optimizer = torch.optim.Adam({dihedral}, lr=learning_rate)
    init_dihedral = dihedral.clone()
    optimal_diheral = dihedral.clone()
    Loss = []
    optimal = float('inf')
    TIMES = int(iteration)
    for i in range(TIMES):
        potential = quadratic_potential(native_distogram, pred_coords)
        Loss.append(potential.item())

        def closure():
            optimizer.zero_grad()
            loss = quadratic_potential(native_distogram, pred_coords)
            loss.backward(retain_graph=True)
            return loss

        if potential < optimal and i > TIMES / 2:
            optimal_diheral = dihedral.clone()
            optimal = potential.item()
        log.write('{:>5}: {:15.5f}'.format(i, potential) + '\n')
        print('{:>5}: {:15.5f}'.format(i, potential))
        # optimizer.zero_grad()
        # potential.backward()
        optimizer.step(closure)
        # grad = dihedral.grad
        # grad[1::3] = 0
        # dihedral.data = dihedral.data - grad * learning_rate
        # dihedral_O[:-1].data = dihedral_O[:-1].data - grad[::3] * learning_rate
        pred_coords = internal_to_coords(first_three_atoms, dihedral, bond_angle, bond_length)
        # dihedral.grad.zero_()
    potential = quadratic_potential(native_distogram, pred_coords)
    Loss.append(potential.item())
    if potential < optimal:
        optimal_diheral = dihedral.clone()
        optimal = potential.item()
    print('{:>5}: {:15.5f}'.format(TIMES, potential))
    log.write('{:>5}: {:15.5f}'.format(TIMES, potential) + '\n')
    print('optimal: {}'.format(optimal))
    log.write('optimal: {}'.format(optimal) + '\n')
    log.close()
    # now_dihedral = dihedral.clone()
    dihedral_O[:-1].data = dihedral_O[:-1].data + (optimal_diheral - init_dihedral)[::3].data
    pred_coords_O = internal_to_coords_withO(first_three_atoms, optimal_diheral, bond_angle, bond_length,
                                             dihedral_O, bond_angle_O, bond_length_O)

    # print(torch.where(abs(delta) > 0.0001, delta, torch.zeros_like(delta)))
    # save pdb
    save_path = os.path.join(os.path.dirname(filename), 'refined_' + str(Loss[-1]) + '.pdb')
    print_pdb(filename, pred_coords_O, save_path)
    dt = datetime.datetime.now().strftime('%m%d_%H%M')
    np.savetxt(os.path.join(os.path.dirname(filename), dt + '_loss.txt'), np.array(Loss))


if __name__ == '__main__':
    start_time = time.time()
    data_path = './data'
    iteration = 1000
    lr = 1
    for target in os.listdir(data_path):
        print(target)
        target = os.path.join(data_path, target)
        if not os.path.isdir(target):
            continue
        refinement(target, lr, iteration)
    # refinement('./data/R1029', lr, iteration)
    end_time = time.time()
    last_time = end_time - start_time
    hours, last_time = divmod(last_time, 3600)
    minutes, seconds = divmod(last_time, 60)
    print('Total time: {:.0f} hours, {:.0f} minutes, {:.4f} seconds'.format(hours, minutes, seconds))
