from torch.utils.data import Dataset
import os
import h5py
import numpy as np


class Completion3D(Dataset):
    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val", "test"]
        split_f = open(os.path.join(root, f"{split}.list"), "r")
        if split in ["train", "val"]:
            self.data = [
                (
                    os.path.join(root, split, "partial", f[:-1] + ".h5"),
                    os.path.join(root, split, "gt", f[:-1] + ".h5"),
                )
                for f in split_f
            ]
        else:
            self.data = [
                (os.path.join(root, split, "partial", f[:-1] + ".h5"), None) for f in split_f
            ]

        self.transform = transform

    def load_pc(self, pc):
        with h5py.File(pc, "r") as f:
            pc = np.array(f["data"])
        return pc

    def __getitem__(self, indx):
        partial, complete = self.data[indx]
        inputs = []
        inputs.append(self.load_pc(partial))
        if complete is not None:
            inputs.append(self.load_pc(complete))
        inputs = self.transform(inputs)
        return inputs

    def __len__(self):
        return len(self.data)


class ModelNet(Dataset):
    def __init__(self, root, n_cls=40, split="train", transform=None):
        assert split in ["train", "test"]
        cls_names = [
            line.rstrip() for line in open(os.path.join(root, f"modelnet{n_cls}_shape_names.txt"))
        ]
        shape_ids = [
            line.rstrip() for line in open(os.path.join(root, f"modelnet{n_cls}_{split}.txt"))
        ]
        self.data = [
            (
                os.path.join(root, "_".join(line.split("_")[0:-1]), f"{line}.txt"),
                cls_names.index("_".join(line.split("_")[0:-1])),
            )
            for line in shape_ids
        ]

        self.transform = transform

    def load_pc(self, pc):
        pc = np.loadtxt(pc, delimiter=",").astype(np.float32)[:, 0:3]
        if self.transform is not None:
            pc = self.transform(pc)
        return pc

    def __getitem__(self, indx):
        pc, target = self.data[indx]
        pc = self.load_pc(pc)
        return pc, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dset = Completion3D(root="/home/sneezygiraffe/data/completion3d")
