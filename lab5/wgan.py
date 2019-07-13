import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import scipy.io as scio
from dataset import pointDataset
from torch.autograd import Variable
import argparse
import os




os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00008, help="learning rate")#0.00005
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=2, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )

    def forward(self, z):
        output = self.model(z)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(50, 1),
        )

    def forward(self, img):
        output = self.model(img)
        return output.squeeze()


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
dataFile = "points.mat"
data = scio.loadmat(dataFile)
training_data = torch.from_numpy(data['xx'])
index = np.arange(training_data.shape[0])


# Loading data
pointData = pointDataset(training_data, index)
dataloader = torch.utils.data.DataLoader(pointData, batch_size=64, shuffle=True)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)
#optimizer_G = torch.optim.SGD(generator.parameters(), lr = 0.1)
#optimizer_D = torch.optim.SGD(discriminator.parameters(), lr = 0.1)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
iterations = 0
fix_noise = torch.randn(1000, 2)
batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):
        # Configure input
        noise = torch.randn(64, 2)
        noise = Variable(noise)
        point = Variable(imgs.float())
        fix_noise = Variable(fix_noise)

        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()


        if (iterations+1) % 100 == 0:
            generator.eval()
            generate_point = generator(fix_noise)
            x = generate_point.data[:, 0].numpy()
            y = generate_point.data[:, 1].numpy()
            x_real = training_data.data[:, 0].numpy()
            y_real = training_data.data[:, 1].numpy()
            x_noise = fix_noise.data[:,0].numpy()
            y_noise = fix_noise.data[:,1].numpy()
            #plt.plot(x_noise,y_noise, 'yx')
            plt.plot(x_real, y_real, 'bx')
            plt.plot(x, y, 'rx')
            plt.axis('equal')
            plt.ylim(-2, 2)   #-10,10
            plt.xlim(-2, 2)   #-10,10
            filename='./wgan1/%d.png' %iterations
            plt.savefig(filename)
            plt.close()

        iterations += 1
        print(iterations)