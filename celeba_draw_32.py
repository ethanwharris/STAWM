import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from memory import Memory

import torchbearer
from torchbearer import Trial, callbacks

import visualise

from torch.distributions import RelaxedBernoulli

MU = torchbearer.state_key('mu')
LOGVAR = torchbearer.state_key('logvar')
STAGES = torchbearer.state_key('stages')
MASKED_TARGET = torchbearer.state_key('masked')


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class CelebDraw(nn.Module):
    def __init__(self, count, glimpse_size, memory_size, output_stages=False):
        super(CelebDraw, self).__init__()

        self.output_stages = output_stages

        self.context = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8, 8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4, 4
            nn.ReLU(True),
            View((-1, 64 * 4 * 4))
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 1, 2),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8, 8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4, 4
            nn.ReLU(True),
            View((-1, 64 * 4 * 4))
        )

        self.decoder = nn.Sequential(
            View((-1, 64, 4, 4)),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3 + 1, 4, 1, 2),  # B, nc + 1, 32, 32
        )

        self.memory = Memory(
            output_inverse=True,
            hidden_size=memory_size,
            memory_size=memory_size,
            glimpse_size=glimpse_size,
            g_down=1024,
            c_down=1024,
            context_net=self.context,
            glimpse_net=self.encoder
        )

        self.count = count
        self.qdown = nn.Linear(1024, memory_size)
        self.soft = nn.LogSoftmax(dim=1)

        self.drop = nn.Dropout(0.3)

        self.mu = nn.Linear(memory_size, 4)
        self.var = nn.Linear(memory_size, 4)

        self.sup = nn.Linear(4, 1024)

        self.onehots = nn.Parameter(torch.eye(count), requires_grad=False)

        if output_stages:
            self.square = visualise.red_square(glimpse_size, width=1).unsqueeze(0).cuda()

    def sample(self, mu, logvar):
        std = logvar.div(2).exp_()
        eps = std.data.new(std.size()).normal_()
        return mu + std * eps

    def forward(self, x, state=None):
        image = x
        canvas = torch.zeros_like(x.data)

        x, context = self.memory.init(image)

        c_data = context.data
        query = F.relu6(self.qdown(c_data))

        mu = []
        var = []
        stages = []
        masks = []

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

            grid = F.affine_grid(inverse, torch.Size([canvas.size(0), canvas.size(1) + 1, canvas.size(2), canvas.size(3)]))
            out = F.grid_sample(out.sigmoid(), grid)

            p = out[:, 0, :, :].unsqueeze(1)
            masks.append(p)

            out = out[:, 1:, :, :]

            dist = RelaxedBernoulli(torch.tensor([2.0]).to(p.device), probs=p)
            p = dist.rsample()

            canvas = canvas * (1 - p)
            out = out * p
            canvas += out

            if self.output_stages:
                square = self.square.clone().repeat(out.size(0), 1, 1, 1)
                square = F.grid_sample(square, grid)

                stage_image = out.data.clone()
                stage_image = stage_image + square
                stage_image = stage_image.clamp(0, 1)
                stages.append(stage_image.unsqueeze(1))

        if state is not None:
            state[torchbearer.Y_TRUE] = image
            state[MU] = torch.stack(mu, dim=1)
            state[LOGVAR] = torch.stack(var, dim=1)
            state[MASKED_TARGET] = state[torchbearer.Y_TRUE].detach() * p.detach()
            if self.output_stages:
                stages.append(image.clone().unsqueeze(1))
                state[STAGES] = torch.cat(stages, dim=1)

        return canvas


def joint_kl_divergence(mu_key, logvar_key, beta=4):
    @callbacks.add_to_loss
    def loss(state):
        mu = state[mu_key]
        logvar = state[logvar_key]

        klds = -0.5 * (logvar.size(1) + logvar.sum(dim=1) - mu.pow(2).sum(dim=1) - logvar.exp().sum(dim=1))
        total_kld = klds.sum(1).mean(0, True)

        return beta * total_kld.item()
    return loss


def draw(count, glimpse_size, memory_size, file, device='cuda'):
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    testset = torchvision.datasets.ImageFolder(root='./cropped_celeba/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=10)

    base_dir = os.path.join('celeba_' + str(memory_size), str(glimpse_size))

    model = CelebDraw(count, glimpse_size, memory_size, output_stages=True)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    from visualise import StagesGrid

    trial = Trial(model, optimizer, nn.MSELoss(reduction='sum'), ['loss'], pass_state=True, callbacks=[
        callbacks.TensorBoardImages(comment=current_time, nrow=10, num_images=20, name='Prediction', write_each_epoch=True,
                                    key=torchbearer.Y_PRED, pad_value=1),
        callbacks.TensorBoardImages(comment=current_time + '_celeb', nrow=10, num_images=20, name='Target', write_each_epoch=False,
                                    key=torchbearer.Y_TRUE, pad_value=1),
        callbacks.TensorBoardImages(comment=current_time + '_celeb_mask', nrow=10, num_images=20, name='Masked Target', write_each_epoch=False,
                                    key=MASKED_TARGET, pad_value=1),
        StagesGrid('celeb_stages.png', STAGES, 20)
    ]).load_state_dict(torch.load(os.path.join(base_dir, file)), resume=False).with_generators(train_generator=testloader, val_generator=testloader).for_train_steps(1).for_val_steps(1).to(device)

    trial.run()  # Evaluate doesn't work with tensorboard in torchbearer, seems to have been fixed in most recent version


def run(count, glimpse_size, memory_size, iteration, device='cuda'):
    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.ImageFolder(root='./cropped_celeba/', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=10)

    base_dir = os.path.join('celeba_' + str(memory_size), str(glimpse_size))

    model = CelebDraw(count, glimpse_size, memory_size)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    call_a = callbacks.TensorBoardImages(comment=current_time, name='Prediction', write_each_epoch=True, key=torchbearer.Y_PRED)
    call_a.on_step_training = call_a.on_step_validation  # Hack to make this log training samples
    call_b = callbacks.TensorBoardImages(comment=current_time + '_celeba', name='Target', write_each_epoch=True,
                                key=torchbearer.Y_TRUE)
    call_b.on_step_training = call_b.on_step_validation  # Hack to make this log training samples

    trial = Trial(model, optimizer, nn.MSELoss(reduction='sum'), ['loss'], pass_state=True, callbacks=[
        joint_kl_divergence(MU, LOGVAR, beta=10),
        callbacks.MostRecent(os.path.join(base_dir, 'iter_' + str(iteration) + '.{epoch:02d}.pt')),
        callbacks.GradientClipping(5),
        call_a,
        call_b
    ]).with_generators(train_generator=trainloader).to(device)

    trial.run(250)


if __name__ == "__main__":
    run(8, 32, 256, 0, device='cuda')
    draw(8, 32, 256, 'iter_0.249.pt')
