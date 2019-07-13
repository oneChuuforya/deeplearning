#windows8系统,python3.7,pytorch-cpu1.1.0
#1160300602于婷

import torch as t
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import datetime
import argparse


# 样本读取线程数
WORKERS = 4

# 网络参数保存文件名
PARAS_FN = 'cifar_Alexnet_params.pkl'

# cifar数据存放位置
ROOT = '/home/yt/PycharmProjects/cifar'

# 目标函数
loss_func = nn.CrossEntropyLoss()

# 最优结果
best_acc = 0

# 定义网络模型
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.cnn = nn.Sequential(
            # 卷积层1，3通道输入，96个卷积核，核大小7*7，步长2，填充2
            # 经过该层图像大小变为32-7+2*2 / 2 +1，15*15
            # 经3*3最大池化，2步长，图像变为15-3 / 2 + 1， 7*7
            nn.Conv2d(3, 96, 7, 2, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            # 卷积层2，96输入通道，256个卷积核，核大小5*5，步长1，填充2
            # 经过该层图像变为7-5+2*2 / 1 + 1，7*7
            # 经3*3最大池化，2步长，图像变为7-3 / 2 + 1， 3*3
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            # 卷积层3，256输入通道，384个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            # 卷积层3，384输入通道，384个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            # 卷积层3，384输入通道，256个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            # 256个feature，每个feature 3*3
            nn.Linear(256*3*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


'''
训练并测试网络
net：网络模型
train_data_load：训练数据集
optimizer：优化器
epoch：第几次训练迭代
log_interval：训练过程中损失函数值和准确率的打印频率
'''
def net_train(net, train_data_load, optimizer, epoch, log_interval,writer):
    net.train()
    begin = datetime.datetime.now()

    # 样本总数
    total = len(train_data_load.dataset)

    # 样本批次训练的损失函数值的和
    train_loss = 0

    # 识别正确的样本数
    ok = 0

    for i, data in enumerate(train_data_load, 0):

        img, label = data


        optimizer.zero_grad()

        outs = net(img)
        loss = loss_func(outs, label)
        loss.backward()
        optimizer.step()

        # 累加损失值和训练样本数
        train_loss += loss.item()
        # total += label.size(0)

        _, predicted = t.max(outs.data, 1)
        # 累加识别正确的样本数
        ok += (predicted == label).sum()

        if (i + 1) % log_interval == 0:
            # 训练结果输出

            # 损失函数均值
            loss_mean = train_loss / (i + 1)

            # 已训练的样本数
            traind_total = (i + 1) * len(label)

            # 准确率
            acc = 100. * ok / traind_total

            # 进度
            progress = 100. * traind_total / total

            writer.add_scalar("Train/loss",loss_mean,epoch)
            writer.add_scalar("Train/acc", acc , epoch)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Acc: {:.6f}'.format(
                epoch, traind_total, total, progress, loss_mean, acc))

    end = datetime.datetime.now()
    print('one epoch spend: ', end - begin)


'''
用测试集检查准确率
'''
def net_test(net, test_data_load, epoch):
    net.eval()
    true = 0
    for i, data in enumerate(test_data_load):
        img, label = data
        outs = net(img)
        _, pre = t.max(outs.data, 1)
        true += (pre == label).sum()

    acc = true.item() * 100. / (len(test_data_load.dataset))
    print('EPOCH:{}, ACC:{}\n'.format(epoch, acc))

    global best_acc
    if acc > best_acc:
        best_acc = acc


def main():
    # 训练超参数设置，可通过命令行设置
    parser = argparse.ArgumentParser(description='PyTorch CIFA10 AlexNet')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='If train the Model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # 图像数值转换，ToTensor源码注释
    # 归一化把[0.0, 1.0]变换为[-1,1], ([0, 1] - 0.5) / 0.5 = [-1, 1]
    transform = tv.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 定义数据集
    train_data = tv.datasets.CIFAR10(root=ROOT, train=True, download=True, transform=transform)
    test_data = tv.datasets.CIFAR10(root=ROOT, train=False, download=False, transform=transform)

    train_load = t.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=WORKERS)
    test_load = t.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=WORKERS)
    writer = SummaryWriter(comment="AlexNet")
    net =AlexNet()
    print(net)

    # 如果不训练，直接加载保存的网络参数进行测试集验证
    if args.no_train:
        net.load_state_dict(t.load(PARAS_FN))
        net_test(net, test_load, 0)
        return

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    start_time = datetime.datetime.now()

    for epoch in range(1, args.epochs + 1):
        net_train(net, train_load, optimizer, epoch, args.log_interval,writer)

        # 每个epoch结束后用测试集检查识别准确度
        net_test(net, test_load, epoch)

    end_time = datetime.datetime.now()

    global best_acc
    print('CIFAR10 pytorch LeNet Train: EPOCH:{}, BATCH_SZ:{}, LR:{}, ACC:{}'.format(args.epochs, args.batch_size, args.lr, best_acc))
    print('train spend time: ', end_time - start_time)

    t.save(net.state_dict(), PARAS_FN)


if __name__ == '__main__':
    main()