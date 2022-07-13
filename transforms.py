import torch
import transforms3d
import numpy as np
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        rnd_val = random.random()
        inputs = inputs if isinstance(inputs, list) else [inputs]
        outs = []
        for input in inputs:
            out = input
            for transform in self.transforms:
                out = transform(out, rnd_val)
            outs.append(out)
        return outs if len(inputs) > 1 else outs[0]


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, pc, rnd_val):
        return torch.from_numpy(pc.copy()).float()


class RandomPermute:
    def __init__(self):
        pass

    def __call__(self, pc, rnd_val):
        perm_indx = np.random.permutation(pc.shape[0])
        pc = pc[perm_indx]
        return pc


class RandomMirror:
    def __init__(self):
        pass

    def __call__(self, pc, rnd_val):
        trans_mat = transforms3d.zooms.zfdir2mat(1)
        trans_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trans_mat)
        trans_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trans_mat)

        if rnd_val <= 0.25:
            trans_mat = np.dot(trans_mat_x, trans_mat)
            trans_mat = np.dot(trans_mat_z, trans_mat)
        elif rnd_val > 0.25 and rnd_val <= 0.5:
            trans_mat = np.dot(trans_mat_x, trans_mat)
        elif rnd_val > 0.5 and rnd_val <= 0.75:
            trans_mat = np.dot(trans_mat_z, trans_mat)

        pc[:, :3] = np.dot(pc[:, :3], trans_mat.T)
        return pc


class RandomScale:
    def __init__(self, low=0.8, high=1.25):
        self.low, self.high = low, high

    def __call__(self, pc, rnd_val):
        scale = rnd_val * (self.high - self.low) + self.low
        pc[:, 0:3] *= scale
        return pc


class RandomShift:
    def __init__(self, range=0.1):
        self.range = range

    def __call__(self, pc, rnd_val):
        delta = rnd_val * self.range
        delta = delta if random.random() < 0.5 else -delta
        pc += delta
        return pc


class Normalize:
    def __init__(self):
        pass

    def __call__(self, pc, rnd_val):
        cen = np.mean(pc, axis=0)
        pc = pc - cen
        range = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / range
        return pc
