import argparse
from datetime import datetime
import random
import numpy as np
import torch
from utils import setup_device, AvgMeter, pbar, print_args, dist_mat, sample_pc
import transforms
import datasets
from torch.utils.data import DataLoader, DistributedSampler
import network
from torch.nn.parallel import DistributedDataParallel
import os
import json
import wandb
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from emd import EMD


class ChamferLoss:
    def __init__(self):
        self.kernel = chamfer_3DDist()

    def __call__(self, pc1, pc2):
        dist1, dist2, _, _ = self.kernel(pc1, pc2)
        return torch.mean(dist1) + torch.mean(dist2)


class PartLoss:
    def __init__(self):
        self.chamfer = ChamferLoss()

    def __call__(self, pc1, pc2, masks):
        mat = dist_mat(pc1, pc2)
        indx = mat.min(dim=2)[-1]
        loss = 0
        for i in range(masks.shape[0]):
            temp = 0
            for j in range(masks.shape[1]):
                p1 = pc1[j][torch.where(masks[i, j] == 1)[0]]
                p2 = pc2[j][indx[j][torch.where(masks[i, j] == 1)[0]]]
                if p1.shape[0] > 0:
                    temp += self.chamfer(p1[None, :, :], p2[None, :, :])
            loss += temp / masks.shape[1]
        return loss


class EmdLoss:
    def __init__(self):
        self.kernel = EMD()

    def __call__(self, pc1, pc2):
        loss = self.kernel(pc1, pc2)
        return loss.mean()


class Trainer:
    def __init__(self, args):
        self.args = args

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.device, local_rank = setup_device(args.dist)
        self.main_thread = True if local_rank == 0 else False
        if self.main_thread:
            print(f"\nsetting up device, distributed = {args.dist}")
        print(f" | {self.device}")

        train_transform = transforms.Compose(
            [
                transforms.RandomPermute(),
                transforms.RandomMirror(),
                transforms.RandomScale(args.scale_low, args.scale_high),
                transforms.ToTensor(),
            ]
        )
        val_transform = transforms.Compose([transforms.ToTensor()])

        if args.dset == "c3d":
            train_dset = datasets.Completion3D(
                args.data_root,
                split="train",
                transform=train_transform,
            )
            val_dset = datasets.Completion3D(
                args.data_root,
                split="val",
                transform=val_transform,
            )
        else:
            raise ValueError(f"args.dset = {args.dset} not implemented")
        if self.main_thread:
            print(f"setting up dataset, train: {len(train_dset)}, val: {len(val_dset)}")
        if args.dist:
            train_sampler = DistributedSampler(train_dset)
            self.train_loader = DataLoader(
                train_dset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.n_workers,
            )
        else:
            self.train_loader = DataLoader(
                train_dset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_workers,
            )
        self.val_loader = DataLoader(
            val_dset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
        )

        if args.net == "pt_comp":
            model = network.CompletionNetwork(
                dims=[int(d) for d in args.dims.split(",")],
                knn_sizes=[int(k) for k in args.knn_sizes.split(",")],
                down_ratio=args.down_ratio,
                block_sizes=[int(b) for b in args.block_sizes.split(",")],
                attn_dim=args.attn_dim,
                pos_hid_dim=args.pos_hid_dim,
                attn_hid_dim=args.attn_hid_dim,
                n_pts=args.n_pts,
                n_regs=args.n_regs,
                rand_reg_p=args.rand_reg_p,
            )
        else:
            raise ValueError(f"args.net = {args.net} not implemented")
        if args.dist:
            torch.set_num_threads(1)
            self.model = DistributedDataParallel(
                model.to(self.device),
                device_ids=[local_rank],
                output_device=local_rank,
            )
        else:
            self.model = model.to(self.device)
        if self.main_thread:
            print(f"# of model parameters: {sum(p.numel() for p in self.model.parameters())/1e6}M")

        self.chamf_criterion = ChamferLoss()
        self.part_criterion = PartLoss()
        self.emd_criterion = EmdLoss()

        if args.optim == "sgd":
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov,
            )
        elif args.optim == "adam":
            self.optim = torch.optim.Adam(
                self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            raise ValueError(f"args.optim = {args.optim} not implemented")

        if self.args.lr_step_mode == "epoch":
            total_steps = args.epochs - args.warmup
        else:
            total_steps = int(args.epochs * len(self.train_loader) - args.warmup)
        if args.warmup > 0:
            for group in self.optim.param_groups:
                group["lr"] = 1e-12 * group["lr"]
        if args.lr_sched == "cosine":
            self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, total_steps)
        elif args.lr_sched == "multi_step":
            milestones = [
                int(milestone) - total_steps for milestone in args.lr_decay_steps.split(",")
            ]
            self.lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optim, milestones=milestones, gamma=args.lr_decay
            )
        else:
            raise ValueError(f"args.lr_sched = {args.lr_sched} not implemented")

        if os.path.exists(os.path.join(args.out_dir, "last.ckpt")):
            if args.resume == False and self.main_thread:
                raise ValueError(
                    f"directory {args.out_dir} already exists, change output directory or use --resume argument"
                )
            ckpt = torch.load(os.path.join(args.out_dir, "last.ckpt"), map_location=self.device)
            model_dict = ckpt["model"]
            if "module" in list(model_dict.keys())[0] and args.dist == False:
                model_dict = {
                    key.replace("module.", ""): value for key, value in model_dict.items()
                }
            self.model.load_state_dict(model_dict)
            self.optim.load_state_dict(ckpt["optim"])
            self.lr_sched.load_state_dict(ckpt["lr_sched"])
            self.start_epoch = ckpt["epoch"] + 1
            if self.main_thread:
                print(
                    f"loaded checkpoint, resuming training expt from {self.start_epoch} to {args.epochs} epochs."
                )
        else:
            if args.resume == True and self.main_thread:
                raise ValueError(
                    f"resume training args are true but no checkpoint found in {args.out_dir}"
                )
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, "args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=4)
            self.start_epoch = 0
            if self.main_thread:
                print(f"starting fresh training expt for {args.epochs} epochs.")
        self.train_steps = self.start_epoch * len(self.train_loader)

        self.log_wandb = False
        self.metric_meter = AvgMeter()
        if self.main_thread:
            self.log_f = open(os.path.join(args.out_dir, "logs.txt"), "w")
            print(f"start file logging @ {os.path.join(args.out_dir, 'logs.txt')}")
            if args.wandb:
                self.log_wandb = True
                run = wandb.init()
                print(f"start wandb logging @ {run.get_url()}")
                self.log_f.write(f"\nwandb url @ {run.get_url()}\n")

    def train_epoch(self):
        self.metric_meter.reset()
        self.model.train()
        for indx, (partial, complete) in enumerate(self.train_loader):
            partial, complete = partial.to(self.device), complete.to(self.device)
            partial = sample_pc(partial, self.args.n_pts)

            pred = self.model(partial)
            pred, reg_loss, masks = pred[0:3], pred[-2], pred[-1]
            res = [p.shape[1] for p in pred]
            complete = sample_pc(complete, res)

            loss = []
            for i in range(len(pred)):
                l = self.chamf_criterion(pred[i].float(), complete[i].float())
                loss.append(l)
            part_loss = self.part_criterion(pred[-1].float(), complete[-1].float(), masks)
            total_loss = (
                self.args.chamf_lam * sum(loss)
                + self.args.reg_lam * reg_loss
                + self.args.part_lam * part_loss
            )

            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

            metrics = {
                "total_train_loss": total_loss.item(),
                "train_chamfer": loss[-1].item(),
                "train_reg_loss": reg_loss.item(),
                "train_part_loss": part_loss.item(),
            }
            self.metric_meter.add(metrics)

            if self.main_thread and indx % self.args.log_every == 0:
                if self.log_wandb:
                    wandb.log({"train step": self.train_steps, **metrics})
                pbar(indx / len(self.train_loader), msg=self.metric_meter.msg())

            if self.args.lr_step_mode == "step":
                if self.train_steps < self.args.warmup and self.args.warmup > 0:
                    self.optim.param_groups[0]["lr"] = (
                        self.train_steps / (self.args.warmup) * self.args.lr
                    )
                else:
                    self.lr_sched.step()

            self.train_steps += 1
        if self.main_thread:
            pbar(1, msg=self.metric_meter.msg())

    @torch.no_grad()
    def eval(self):
        self.metric_meter.reset()
        self.model.eval()
        n_vis = 0
        for indx, (partial, complete) in enumerate(self.val_loader):
            partial, complete = partial.to(self.device), complete.to(self.device)
            partial = sample_pc(partial, self.args.n_pts)

            pred = self.model(partial)
            pred, reg_loss, masks = pred[0:3], pred[-2], pred[-1]
            res = [p.shape[1] for p in pred]
            complete = sample_pc(complete, res)

            loss = []
            for i in range(len(pred)):
                l = self.chamf_criterion(pred[i].float(), complete[i].float())
                loss.append(l)
            part_loss = self.part_criterion(pred[-1].float(), complete[-1].float(), masks)
            total_loss = (
                self.args.chamf_lam * sum(loss)
                + self.args.reg_lam * reg_loss
                + self.args.part_lam * part_loss
            )

            emd_loss = self.emd_criterion(pred[-1].float(), complete[-1].float())
            metrics = {
                "total_val_loss": total_loss.item(),
                "val_chamfer": loss[-1].item(),
                "val_reg_loss": reg_loss.item(),
                "val_part_loss": part_loss.item(),
                "val_emd_loss": emd_loss.item(),
            }
            self.metric_meter.add(metrics)

            if self.main_thread:
                if self.log_wandb and random.random() < 0.1 and n_vis < self.args.n_vis:
                    vis_indx = random.randint(0, complete[-1].shape[0] - 1)
                    vis = []
                    vis.append(complete[-1][vis_indx].cpu().detach().numpy())
                    temp = pred[-1][vis_indx].cpu().detach().numpy()
                    temp[:, 0] += 2
                    vis.append(temp)
                    temp = partial[vis_indx].cpu().detach().numpy()
                    temp[:, 0] += 4
                    vis.append(temp)
                    wandb.log(
                        {f"sample_{n_vis}": wandb.Object3D(np.concatenate(vis, axis=0), axis=0)}
                    )
                    n_vis += 1

                if indx % self.args.log_every == 0:
                    pbar(indx / len(self.val_loader), msg=self.metric_meter.msg())

        if self.main_thread:
            pbar(1, msg=self.metric_meter.msg())

    def train(self):
        best_train, best_val = float("inf"), float("inf")

        for epoch in range(self.start_epoch, self.args.epochs):
            if self.main_thread:
                print(f"\nepoch: {epoch}")
                print("---------------")

            self.train_epoch()

            if self.main_thread:
                train_metrics = self.metric_meter.get()
                if train_metrics["train_chamfer"] < best_train:
                    print(
                        "\x1b[34m"
                        + f"train chamfer improved from {round(best_train, 5)} to {round(train_metrics['train_chamfer'], 5)}"
                        + "\033[0m"
                    )
                    best_train = train_metrics["train_chamfer"]
                msg = f"epoch: {epoch}, last train: {round(train_metrics['train_chamfer'], 5)}, best train: {round(best_train, 5)}"

                val_metrics = {}
                if epoch % self.args.eval_every == 0:
                    self.eval()
                    val_metrics = self.metric_meter.get()
                    if val_metrics["val_chamfer"] < best_val:
                        print(
                            "\x1b[33m"
                            + f"val chamfer improved from {round(best_val, 5)} to {round(val_metrics['val_chamfer'], 5)}"
                            + "\033[0m"
                        )
                        best_val = val_metrics["val_chamfer"]
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.args.out_dir, f"best.ckpt"),
                        )
                    msg += f", last val: {round(val_metrics['val_chamfer'], 5)}, best val: {round(best_val, 5)}"

                print(msg)
                self.log_f.write(msg + f", lr: {round(self.optim.param_groups[0]['lr'], 5)}\n")
                self.log_f.flush()

                if self.log_wandb:
                    train_metrics = {"epoch " + key: value for key, value in train_metrics.items()}
                    val_metrics = {"epoch " + key: value for key, value in val_metrics.items()}
                    wandb.log(
                        {
                            "epoch": epoch,
                            **train_metrics,
                            **val_metrics,
                            "lr": self.optim.param_groups[0]["lr"],
                        }
                    )

                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                        "lr_sched": self.lr_sched.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.args.out_dir, "last.ckpt"),
                )

            if self.args.lr_step_mode == "epoch":
                if epoch <= self.args.warmup and self.args.warmup > 0:
                    self.optim.param_groups[0]["lr"] = epoch / self.args.warmup * self.args.lr
                else:
                    self.lr_sched.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_dir",
        type=str,
        default=f"output/{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
        help="path to output directory [def: output/year-month-date_hour-minute]",
    )
    parser.add_argument("--seed", type=int, default=42, help="set experiment seed [def: 42]")
    parser.add_argument(
        "--dist", action="store_true", help="start distributed training [def: false]"
    )
    parser.add_argument(
        "--n_pts", type=int, default=1024, help="set number of input points [def: 1024]"
    )
    parser.add_argument(
        "--scale_low", type=float, default=0.8, help="scale augmentation range start [def: 0.8]"
    )
    parser.add_argument(
        "--scale_high", type=float, default=1.25, help="scale augmentation range end [def: 1.25]"
    )
    parser.add_argument(
        "--shift_range", type=float, default=0.1, help="shift augmentation range [def: 0.1]"
    )
    parser.add_argument("--dset", type=str, default="c3d", help="dataset name [def: m40]")
    parser.add_argument("--data_root", type=str, required=True, help="dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size [def: 16]")
    parser.add_argument(
        "--n_workers", type=int, default=4, help="number of workers for dataloading [def: 4]"
    )
    parser.add_argument("--net", type=str, default="pt_comp", help="network name [def: pt_cls]")
    parser.add_argument(
        "--dims",
        type=str,
        default="64,128,256",
        help="network dim sizes [def: 64,128,256]",
    )
    parser.add_argument(
        "--knn_sizes", type=str, default="32,32,32", help="network knn sizes [def: 32,32,32]"
    )
    parser.add_argument(
        "--down_ratio", type=int, default=4, help="network downsampling ratio [def: 4]"
    )
    parser.add_argument(
        "--block_sizes", type=str, default="2,2,2", help="network block sizes [def: 2,2,2]"
    )
    parser.add_argument(
        "--attn_dim", type=int, default=64, help="model downsampling ratio [def: 4]"
    )
    parser.add_argument(
        "--pos_hid_dim", type=int, default=64, help="model downsampling ratio [def: 4]"
    )
    parser.add_argument(
        "--attn_hid_dim", type=int, default=256, help="model downsampling ratio [def: 4]"
    )
    parser.add_argument("--n_regs", type=int, default=4, help="number of regions [def: 4]")
    parser.add_argument(
        "--rand_reg_p", type=float, default=0.1, help="random region probability [def: 0.1]"
    )
    parser.add_argument(
        "--rand_reg_epochs", type=int, default=10, help="random region routing epochs [def: 10]"
    )
    parser.add_argument("--optim", type=str, default="adam", help="optimizer name [def: adam]")
    parser.add_argument("--lr", type=float, default=0.0001, help="sgd learning rate [def: 0.0001]")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="sgd optimizer momentum [def: 0.9]"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="sgd optimizer weight decay [def: 0.0001]"
    )
    parser.add_argument(
        "--nesterov", type=bool, default=False, help="nesterov in sgd optim [def: false]"
    )
    parser.add_argument(
        "--lr_step_mode",
        type=str,
        default="epoch",
        help="choose lr step mode, one of [epoch, step] [def: epoch]",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="lr warmup in epochs/steps based on epoch step mode [def: 0]",
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs [def: 200]")
    parser.add_argument(
        "--lr_sched", type=str, default="multi_step", help="lr scheduler name [def: multi_step]"
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=str,
        default="120,160",
        help="multi step lr scheduler milestones [def: 120, 160]",
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.1,
        help="multi step lr scheduler decay gamma [def: 0.1]",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training from checkpoint [def: false]"
    )
    parser.add_argument("--wandb", action="store_true", help="start wandb logging [def: false]")
    parser.add_argument("--eval_every", type=int, default=1, help="eval frequency [def: 1]")
    parser.add_argument("--log_every", type=int, default=1, help="logging frequency [def: 1]")
    parser.add_argument(
        "--chamf_lam", type=float, default=1e3, help="chamfer loss lambda [def: 1e3]"
    )
    parser.add_argument("--reg_lam", type=float, default=1, help="region loss lambda [def: 1]")
    parser.add_argument("--part_lam", type=float, default=1e2, help="part loss lambda [def: 1e2]")
    parser.add_argument(
        "--n_vis", type=int, default=5, help="num of pcs to visualize during eval [def: 5]"
    )
    args = parser.parse_args()
    print_args(args)

    trainer = Trainer(args)
    trainer.train()

    if args.dist:
        torch.distributed.destroy_process_group()
