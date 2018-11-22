import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from memory import Memory

import tb_modules as tm

import torchbearer
from torchbearer import Trial, callbacks

import visualise

MU = torchbearer.state_key('mu')
LOGVAR = torchbearer.state_key('logvar')
STAGES = torchbearer.state_key('stages')


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        torch.nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class InverseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(InverseBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        torch.nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class CifarDraw(nn.Module):
    def __init__(self, count, memory_size, output_stages=False):
        super(CifarDraw, self).__init__()

        self.output_stages = output_stages

        self.context = nn.Sequential(
            Block(3, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            Block(32, 32, 4, 2, 1),  # B,  32, 8, 8
            nn.ReLU(True),
            Block(32, 32, 4, 2, 1),  # B,  64,  4, 4
            nn.ReLU(True),
            View((-1, 32 * 4 * 4))
        )

        self.encoder = nn.Sequential(
            Block(3, 32, 4, 1, 2),  # B,  32, 16, 16
            nn.ReLU(True),
            Block(32, 64, 4, 2, 1),  # B,  32, 8, 8
            nn.ReLU(True),
            Block(64, 128, 4, 2, 1),  # B,  64,  4, 4
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1),  # B,  64,  2, 2
            nn.ReLU(True),
            View((-1, 128 * 2 * 2))
        )

        self.decoder = nn.Sequential(
            View((-1, 128, 2, 2)),
            InverseBlock(128, 128, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            InverseBlock(128, 64, 4, 2, 1),  # B,  32, 8, 8
            nn.ReLU(True),
            InverseBlock(64, 32, 4, 2, 1, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            InverseBlock(32, 3, 4, 1, 2),  # B, nc, 16, 16
        )

        self.memory = Memory(output_inverse=True, hidden_size=memory_size, memory_size=memory_size, glimpse_size=16, g_down=512, c_down=512, context_net=self.context, glimpse_net=self.encoder)

        self.count = count

        self.drop = nn.Dropout(0.3)

        self.qdown = nn.Linear(512, memory_size)

        self.mu = nn.Linear(memory_size, 32)
        self.var = nn.Linear(memory_size, 32)

        self.sup = nn.Linear(32, 512)

        if output_stages:
            self.square = visualise.red_square(16, width=1).unsqueeze(0).cuda()

    def sample(self, mu, logvar):
        if self.training:
            std = logvar.div(2).exp_()
            eps = std.data.new(std.size()).normal_()
            return mu + std * eps
        else:
            return mu

    def forward(self, x, state=None):
        image = x
        canvas = torch.zeros_like(x.data)

        x, context = self.memory.init(image)

        c_data = context.data
        query = F.relu6(self.qdown(c_data))

        mu = []
        var = []
        stages = []

        for i in range(self.count):
            x, inverse = self.memory.glimpse(x, image)
            out = self.memory(query)
            o_mu = self.mu(out)
            o_var = self.var(out)
            mu.append(o_mu)
            var.append(o_var)
            out = self.sample(o_mu, o_var)
            out = F.relu(self.sup(out))
            out = self.decoder(out)

            inverse = inverse.view(out.size(0), 2, 3)

            grid = F.affine_grid(inverse, canvas.size())
            out = F.grid_sample(out, grid)

            canvas += out

            if self.output_stages:
                square = self.square.clone().repeat(out.size(0), 1, 1, 1)
                square = F.grid_sample(square, grid)

                stage_image = canvas.data.clone().sigmoid()
                stage_image = stage_image + square
                stage_image = stage_image.clamp(0, 1)
                stages.append(stage_image.unsqueeze(1))

        if state is not None:
            state[torchbearer.Y_TRUE] = image
            state[MU] = torch.cat(mu, dim=1)
            state[LOGVAR] = torch.cat(var, dim=1)
            if self.output_stages:
                stages.append(image.clone().unsqueeze(1))
                state[STAGES] = torch.cat(stages, dim=1)

        return canvas.sigmoid()


def draw(count, memory_size, file, device='cuda'):
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=10)

    base_dir = os.path.join('cifar_' + str(memory_size), "16")

    model = CifarDraw(count, memory_size, output_stages=True)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    from visualise import StagesGrid

    trial = Trial(model, optimizer, nn.MSELoss(reduction='sum'), ['acc', 'loss'], pass_state=True, callbacks=[
        callbacks.TensorBoardImages(comment=current_time, name='Prediction', write_each_epoch=True,
                                    key=torchbearer.Y_PRED, pad_value=1, nrow=16),
        callbacks.TensorBoardImages(comment=current_time + '_cifar', name='Target', write_each_epoch=False,
                                    key=torchbearer.Y_TRUE, pad_value=1, nrow=16),
        StagesGrid('cifar_stages.png', STAGES, 20)
    ]).load_state_dict(torch.load(os.path.join(base_dir, file)), resume=False).with_generators(train_generator=testloader, val_generator=testloader).for_train_steps(1).to(device)

    trial.run()  # Evaluate doesn't work with tensorboard in torchbearer, seems to have been fixed in most recent version


def run(count, memory_size, iteration, device='cuda'):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=10)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=10)

    base_dir = os.path.join('cifar_' + str(memory_size), "16")

    model = CifarDraw(count, memory_size)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    trial = Trial(model, optimizer, nn.MSELoss(reduction='sum'), ['acc', 'loss'], pass_state=True, callbacks=[
        tm.kl_divergence(MU, LOGVAR, beta=2),
        callbacks.MultiStepLR([50, 90]),
        callbacks.MostRecent(os.path.join(base_dir, 'iter_' + str(iteration) + '.{epoch:02d}.pt')),
        callbacks.GradientClipping(5),
        callbacks.TensorBoardImages(comment=current_time, name='Prediction', write_each_epoch=True,
                                    key=torchbearer.Y_PRED),
        callbacks.TensorBoardImages(comment=current_time + '_cifar', name='Target', write_each_epoch=False,
                                    key=torchbearer.Y_TRUE),
    ]).with_generators(train_generator=trainloader, val_generator=testloader).for_val_steps(5).to(device)

    trial.run(100)


if __name__ == "__main__":
    run(8, 256, 0)
    draw(8, 256, 'iter_0.99.pt')
