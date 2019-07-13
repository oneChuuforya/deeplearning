import torch
import torch.nn as nn


class pointGeneratorLayer(nn.Module):
    def __init__(self):
        super(pointGeneratorLayer, self).__init__()
        self.weight_1 = nn.Parameter(torch.zeros(2, 2))

    def forward(self, x):
        output_1 = torch.mm(x, self.weight_1)
        return output_1


class pointDiscriminatorLayer(nn.Module):
    def __init__(self):
        super(pointDiscriminatorLayer, self).__init__()
        self.weight_1 = nn.Parameter(torch.zeros(2, 1))
        # self.weight_2 = nn.Parameter(torch.zeros(10, 1))

    def forward(self, x):
        # unfortunately we don't have automatic broadcasting yet
        output_1 = torch.mm(x, self.weight_1)
        # output_2 = torch.mm(output_1, self.weight_2)
        return output_1


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50,2),
        )

    def forward(self, x):
        output = self.projector(x)

        return output



class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.projector(x)
        output = self.sigmoid(output)
        return output.squeeze()