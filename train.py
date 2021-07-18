import torch
import dgl
from model import GraphTransformerNet
import torch.nn.functional as F
import torch.nn as nn
from data import GraphDataset, collate
from torch.utils.data import DataLoader
import datetime

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def main(train_dataloader, validate_dataloader, model, optimizer, loss_fn, epochs, batch_size):
    model = model.to(device)
    model.train()
    size = len(train_dataloader.dataset)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        for batch_id, (graphs, labels) in enumerate(train_dataloader):
            if torch.any(torch.isnan(graphs.ndata['h'])) \
                    or torch.any(torch.isnan(graphs.edata['h'])) \
                    or torch.any(torch.isnan(labels)):
                continue
            graphs = graphs.to(device)
            labels = labels.to(device)
            pred = model(graphs, graphs.ndata['h'], graphs.edata['h'])
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_id % 10 == 0:
                loss, current = loss.item(), batch_id * batch_size
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:5d}]")
    torch.save(model.state_dict(), 'model_weight.pth')


def train_loop(dataloader, model, loss_fn, optimizer, epoch, epochs):
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    for batch_id, (graphs, labels) in enumerate(dataloader):
        if torch.any(torch.isnan(graphs.ndata['h'])) \
                or torch.any(torch.isnan(graphs.edata['h'])) \
                or torch.any(torch.isnan(labels)):
            continue
        graphs = graphs.to(device)
        labels = labels.to(device)
        pred = model(graphs, graphs.ndata['h'], graphs.edata['h'])
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()
        if batch_id % 5 == 0:
            loss, current = loss.item(), batch_id * graphs.batch_size
            print(f"loss: {loss:>7f}  [Epoch {epoch + 1:>3d}/{epochs:<3d}] [{current:>5d}/{size:<5d}]")
    return train_loss / num_batches


def validate_loop(dataloader, model, loss_fn, epoch, epochs):
    model.eval()
    num_batches = len(dataloader)
    val_loss = 0
    with torch.no_grad():
        for graphs, labels in dataloader:
            if torch.any(torch.isnan(graphs.ndata['h'])) \
                    or torch.any(torch.isnan(graphs.edata['h'])) \
                    or torch.any(torch.isnan(labels)):
                continue
            graphs = graphs.to(device)
            labels = labels.to(device)
            pred = model(graphs, graphs.ndata['h'], graphs.edata['h'])
            val_loss += loss_fn(pred, labels).item()

    val_loss /= num_batches
    print(f"val_loss: {val_loss:>8f} [Epoch {epoch + 1:>3d}/{epochs:<3d}]  \n")
    return val_loss


if __name__ == '__main__':
    dataset = GraphDataset('traindata')
    train_size = int(0.8 * len(dataset))
    train_set, validate_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    print("train: {}\nvalida: {}".format(len(train_set), len(validate_set)))
    batch_size = 4
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    validate_dataloader = DataLoader(validate_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = GraphTransformerNet(46, 12, 128, 1, 6).to(device)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1, verbose=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
    criterion = nn.MSELoss()
    epochs = 2
    dt = datetime.datetime.now().strftime('%m%d_%H%M')
    log_loss = open("loss_{}.txt".format(dt), 'w')
    log_loss.write('train_loss\tval_loss\n')
    for epoch in range(epochs):
        train_loss = train_loop(train_dataloader, model, criterion, optimizer, epoch, epochs)
        val_loss = validate_loop(validate_dataloader, model, criterion, epoch, epochs)
        scheduler.step()
        log_loss.write("{:<7.5f}\t{:<7.5f}\n".format(train_loss, val_loss))
    torch.save(model.state_dict(), 'models/model_{}.pth'.format(dt))
