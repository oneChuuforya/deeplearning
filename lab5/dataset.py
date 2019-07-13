from torch.utils.data import Dataset
class pointDataset(Dataset):
    def __init__(self, data, lebl, transform=None):
        self.data = data
        self.labels = lebl
        self.transform = transform

    def __getitem__(self, index):
        single_label = self.labels[index]
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)

        return (data, single_label)

    def __len__(self):
        return len(self.labels)