import os
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from GCN import GCN

os.environ['TORCH'] = torch.__version__
os.environ['PYTHONWARNINGS'] = "ignore"

if __name__ == '__main__':

    # Import dataset
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    # Train model
    model = GCN(dataset, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()


    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        the_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        the_loss.backward()
        optimizer.step()
        return the_loss


    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        prediction = out.argmax(dim=1)
        test_correct = prediction[data.test_mask] == data.y[data.test_mask]
        my_test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        return my_test_acc


    for epoch in range(1, 101):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    # Evaluate model
    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')