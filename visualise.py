import torch
import torchvision.utils as utils
from torchbearer.callbacks import Callback


def red_square(s, width=2):
    canvas = torch.zeros(3, s, s)

    canvas[0, :, :width] = 1.0
    canvas[0, :width] = 1.0
    canvas[0, :, s - width] = 1.0
    canvas[0, s - width] = 1.0

    return canvas


class StagesGrid(Callback):
    def __init__(self, path, key, num_images):
        self.path = path
        self.key = key
        self.num_images = num_images

        self.done=False

    def on_step_validation(self, state):
        if not self.done:
            data = state[self.key].data

            if len(data.size()) == 4:
                data = data.unsqueeze(2)

            if state['t'] == 0:
                remaining = self.num_images if self.num_images < data.size(0) else data.size(0)

                self._data = data[:remaining].cpu()
            else:
                remaining = self.num_images - self._data.size(0)

                if remaining > data.size(0):
                    remaining = data.size(0)

                self._data = torch.cat((self._data, data[:remaining].cpu()), dim=0)

            if self._data.size(0) >= self.num_images:
                image_data = [] #torch.zeros(self._data.size(0) * self._data.size(1), self._data.size(2), self._data.size(3), self._data.size(4))

                for i in range(self._data.size(1)):
                    for j in range(self._data.size(0)):
                        image_data.append(self._data[j][i].unsqueeze(0))

                image_data = torch.cat(image_data, dim=0)

                utils.save_image(image_data, self.path, nrow=self.num_images, pad_value=1)
                self.done = True


class WrongImages(Callback):
    def __init__(self,
                 name,
                 key='y_pred',
                 pred='class',
                 target='y_true',
                 write_each_epoch=True,
                 num_images=50,
                 nrow=10,
                 padding=2,
                 normalize=False,
                 range=None,
                 scale_each=False,
                 pad_value=0):
        self.name = name
        self.key = key
        self.pred = pred
        self.target = target
        self.write_each_epoch = write_each_epoch
        self.num_images = num_images
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.range = range
        self.scale_each = scale_each
        self.pad_value = pad_value

        self.done = False

        self.preds = []
        self.targets = []

        self._data = None

    def on_step_validation(self, state):

        pred = state[self.pred]
        target = state[self.target]

        if not self.done and not torch.max(pred, 1)[1][0] == target[0]:

            self.preds.append(torch.max(pred, 1)[1][0].item())
            self.targets.append(target[0].item())

            data = state[self.key].data.clone()

            if len(data.size()) == 3:
                data = data.unsqueeze(1)

            if self._data is None:
                remaining = self.num_images if self.num_images < data.size(0) else data.size(0)

                self._data = data[:remaining].cpu()
            else:
                remaining = self.num_images - self._data.size(0)

                if remaining > data.size(0):
                    remaining = data.size(0)

                self._data = torch.cat((self._data, data[:remaining].cpu()), dim=0)

            if self._data.size(0) >= self.num_images:
                utils.save_image(
                    self._data,
                    self.name,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    range=self.range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value
                )
                self.done = True

    def on_end_epoch(self, state):
        if self.write_each_epoch:
            self.done = False