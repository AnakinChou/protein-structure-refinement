import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import dgl
import os


class GraphDataset(Dataset):
    def __init__(self, data_dir):
        self.fileList = self.get_files(data_dir)
        self.boundaries = torch.linspace(2, 20, 37)  # [2,2.5,...,20] num_bins=38 left closed and right open

    def get_files(self, data_dir):
        folders = os.listdir(data_dir)
        fileList = []
        for folder in folders:
            files = os.listdir(os.path.join(data_dir, folder))
            for file in files:
                absname = os.path.join(data_dir, folder, file)
                if not absname.endswith('.bin'):
                    continue
                fileList.append(absname)
        return fileList

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        g, label_dict = dgl.load_graphs(self.fileList[idx])
        return g[0], torch.bucketize(label_dict['label'], self.boundaries, right=True)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batch = dgl.batch(graphs)
    batch_label = labels[0]
    for i in range(1, len(labels)):
        batch_label = torch.cat([batch_label, labels[i]])
    return batch, batch_label


if __name__ == '__main__':
    train_data = GraphDataset('traindata')
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate)
    for gs, ls in train_dataloader:
        print(len(gs))
        print(ls.shape)
