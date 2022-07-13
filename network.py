import torch.nn as nn
from utils import dist_mat, index, fps, index
import torch
import random


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features,
        eps=0.00001,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        super(BatchNorm1d, self).__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=0.00001,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        super(BatchNorm2d, self).__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = super().forward(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


def query_group(x, pc1, pc2, grp_size, cat_pc=False):
    mat = dist_mat(pc1, pc2)
    knn_indx = torch.topk(mat, dim=2, k=grp_size, largest=False, sorted=True)[1]
    knn_pc = index(pc2, knn_indx)
    knn_pc = knn_pc - pc1[:, :, None, :]
    knn_x = index(x, knn_indx)

    if cat_pc:
        return torch.cat([knn_pc, knn_x], dim=-1)
    else:
        return knn_x


class EncoderAttn(nn.Module):
    def __init__(self, in_dim, dim, pos_hid_dim, attn_hid_dim, knn_size):
        super(EncoderAttn, self).__init__()
        self.pre_fc = nn.Linear(in_dim, dim)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.pos_fc = nn.Sequential(
            nn.Linear(3, pos_hid_dim),
            BatchNorm2d(pos_hid_dim),
            nn.ReLU(),
            nn.Linear(pos_hid_dim, dim),
        )

        self.attn_fc = nn.Sequential(
            nn.Linear(dim, attn_hid_dim),
            BatchNorm2d(attn_hid_dim),
            nn.ReLU(),
            nn.Linear(attn_hid_dim, dim),
        )

        self.post_fc = nn.Linear(dim, in_dim)
        self.knn_size = knn_size

    def forward(self, x):
        x, pc = x
        residue = x
        x = self.pre_fc(x)

        q, k, v = self.q(x), self.k(x), self.v(x)
        k = query_group(k, pc, pc, self.knn_size, cat_pc=True)
        v = query_group(v, pc, pc, self.knn_size, cat_pc=False)

        pos, k = k[:, :, :, 0:3], k[:, :, :, 3:]
        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        attn = torch.softmax(attn, dim=-2)
        x = ((v + pos) * attn).sum(2)

        x = self.post_fc(x) + residue
        return x, pc


class EncoderGrp(nn.Module):
    def __init__(self, in_dim, dim, down_ratio, knn_size):
        super(EncoderGrp, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 + in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.down_ratio = down_ratio
        self.knn_size = knn_size

    def forward(self, x):
        x, pc = x
        if self.down_ratio != 1:
            pc_fps = fps(pc, pc.shape[1] // self.down_ratio)
        else:
            pc_fps = pc
        x = query_group(x, pc_fps, pc, self.knn_size, cat_pc=True)
        x = self.fc(x)
        x = x.max(dim=-2)[0]
        return x, pc_fps


class Encoder(nn.Module):
    def __init__(
        self, dims, knn_sizes, down_ratio, block_sizes, attn_dim, pos_hid_dim, attn_hid_dim
    ):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList()
        in_dim = 3
        for i, (dim, knn_size, block_size) in enumerate(zip(dims, knn_sizes, block_sizes)):
            fc = EncoderGrp(
                in_dim=in_dim, dim=dim, down_ratio=1 if i == 0 else down_ratio, knn_size=knn_size
            )
            attn = [
                EncoderAttn(
                    in_dim=dim,
                    dim=attn_dim,
                    pos_hid_dim=pos_hid_dim,
                    attn_hid_dim=attn_hid_dim,
                    knn_size=knn_size,
                )
                for _ in range(block_size)
            ]
            self.blocks.append(nn.Sequential(fc, *attn))
            in_dim = dim

    def forward(self, x):
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x)
        return out


class DecoderGrp(nn.Module):
    def __init__(self, in_dim, dim, down_ratio, knn_size):
        super(DecoderGrp, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 + in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.down_ratio = down_ratio
        self.knn_size = knn_size

    def forward(self, x):
        dec_x, dec_pc, enc_x, enc_pc = x
        x, pc = dec_x, dec_pc
        if self.down_ratio != 1:
            pc_fps = fps(pc, pc.shape[1] // self.down_ratio)
        else:
            pc_fps = pc
        x = query_group(x, pc_fps, pc, self.knn_size, cat_pc=True)
        x = self.fc(x)
        x = x.max(dim=-2)[0]
        return x, pc_fps, enc_x, enc_pc


class DecoderAttn(nn.Module):
    def __init__(self, in_dim1, in_dim2, dim, pos_hid_dim, attn_hid_dim, knn_size, last=False):
        super(DecoderAttn, self).__init__()
        self.pre_fc1 = nn.Linear(in_dim1, dim)
        self.pre_fc2 = nn.Linear(in_dim2, dim)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.pos_fc = nn.Sequential(
            nn.Linear(3, pos_hid_dim),
            BatchNorm2d(pos_hid_dim),
            nn.ReLU(),
            nn.Linear(pos_hid_dim, dim),
        )

        self.attn_fc = nn.Sequential(
            nn.Linear(dim, attn_hid_dim),
            BatchNorm2d(attn_hid_dim),
            nn.ReLU(),
            nn.Linear(attn_hid_dim, dim),
        )

        self.post_fc1 = nn.Linear(dim, in_dim1)
        self.post_fc2 = nn.Identity()
        if not last:
            self.post_fc2 = nn.Linear(dim, in_dim2)

        self.knn_size = knn_size

    def forward(self, x):
        dec_x, dec_pc, enc_x, enc_pc = x
        residue = dec_x
        dec_x = self.pre_fc1(dec_x)
        enc_x = self.pre_fc2(enc_x)

        pc = torch.cat([dec_pc, enc_pc], dim=1)
        x = torch.cat([dec_x, enc_x], dim=1)
        fps_indx = fps(pc, dec_pc.shape[1], indx_only=True)
        pc = index(pc, fps_indx)
        x = index(x, fps_indx)

        q, k, v = self.q(dec_x), self.k(x), self.v(x)
        k = query_group(k, dec_pc, pc, self.knn_size, cat_pc=True)
        v = query_group(v, dec_pc, pc, self.knn_size, cat_pc=False)

        pos, k = k[:, :, :, 0:3], k[:, :, :, 3:]
        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        attn = torch.softmax(attn, dim=-2)
        x = ((v + pos) * attn).sum(2)

        dec_x = self.post_fc1(x) + residue
        enc_x = self.post_fc2(x)
        return dec_x, dec_pc, enc_x, enc_pc


def grid(up_ratio, grid_size=0.2):
    sqrted = int(up_ratio ** 0.5) + 1
    for i in range(1, sqrted + 1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio // i
            break

    grid_x = torch.linspace(-grid_size, grid_size, steps=num_x)
    grid_y = torch.linspace(-grid_size, grid_size, steps=num_y)

    x, y = torch.meshgrid(grid_x, grid_y)
    grid = torch.stack([x, y], dim=-1).view(-1, 2)
    return grid


class RegionGrouping(nn.Module):
    def __init__(self, in_dim, n_regs, out_dim, rand_prob):
        super(RegionGrouping, self).__init__()
        self.occ = nn.Linear(in_dim, n_regs)
        # self.g_fc = nn.Sequential(
        #     nn.Linear(in_dim, out_dim),
        #     nn.ReLU(),
        #     nn.Linear(out_dim, out_dim),
        #     nn.ReLU(),
        # )
        self.reg_fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        self.rand_prob = rand_prob
        self.n_regs = n_regs

    def forward(self, x, g_vec, return_masks=False):
        occ = self.occ(x)
        occ = torch.softmax(occ, dim=-1)
        reg_logit, reg_indx = occ.topk(k=2, dim=-1)
        if self.training and random.random() < self.rand_prob:
            reg_logit = reg_logit[:, :, [1]]
            reg_indx = reg_indx[:, :, [1]]
        else:
            reg_logit = reg_logit[:, :, [0]]
            reg_indx = reg_indx[:, :, [0]]
        occ_m = torch.zeros_like(occ)
        occ_m.scatter_(2, reg_indx, reg_logit)
        reg_dist_loss = (occ_m.mean(dim=1)**2).sum(dim=1).mean()

        out = torch.zeros_like(x)
        masks = []
        for i in range(self.n_regs):
            mask = reg_indx[:, :, 0] == i
            reg_vec = (self.reg_fc(x * mask[:, :, None])).max(dim=1)[0]
            out = out + mask[:, :, None] * reg_vec[:, None, :]
            if return_masks:
                masks.append(mask[None, :, :])
        #g = self.g_fc(x).max(dim=1)[0]
        g = g_vec
        out = torch.cat([x, out, g[:, None, :].repeat(1, x.shape[1], 1)], dim=2)
        if return_masks:
            return out, reg_dist_loss, torch.cat(masks, dim=0)
        else:
            return out, reg_dist_loss


class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        n_pts,
        up_ratio,
        knn_sizes,
        block_sizes,
        attn_dim,
        pos_hid_dim,
        attn_hid_dim,
        n_regs,
        rand_reg_p,
    ):
        super(Decoder, self).__init__()
        self.g_fc1 = nn.Sequential(
            nn.Linear(dims[-1], dims[-1]),
            nn.ReLU(),
        )
        self.g_fc2 = nn.Sequential(
            nn.Linear(dims[-1], dims[-2]),
            nn.ReLU(),
        )
        self.g_fc3 = nn.Sequential(
            nn.Linear(dims[-2], dims[-3]),
            nn.ReLU(),
        )

        self.g_pc3 = nn.Linear(dims[-3], n_pts // (up_ratio ** 2) * 3)
        self.pre_fc3 = nn.Sequential(
            nn.Linear(3, dims[-3]), nn.ReLU(), nn.Linear(dims[-3], dims[-3])
        )
        fc = DecoderGrp(in_dim=dims[-3], dim=dims[-3], down_ratio=1, knn_size=knn_sizes[-3])
        attn = [
            DecoderAttn(
                in_dim1=dims[-3],
                in_dim2=dims[-1],
                dim=attn_dim,
                pos_hid_dim=pos_hid_dim,
                attn_hid_dim=attn_hid_dim,
                knn_size=knn_sizes[-3],
                last=(i+1)==block_sizes[-3]
            )
            for i in range(block_sizes[-3])
        ]
        self.dec3 = nn.Sequential(fc, *attn)
        self.post_fc3 = nn.Linear(dims[-3], 3)

        self.grid2 = grid(up_ratio, 0.2).cuda().contiguous()
        self.g_pc21 = nn.Sequential(
            nn.Linear(dims[-2], dims[-2]),
            nn.ReLU(),
        )
        self.g_pc22 = nn.Sequential(
            nn.Linear(dims[-2] + 2 + 3, dims[-2]),
            nn.ReLU(),
            nn.Linear(dims[-2], dims[-2]),
            nn.ReLU(),
            nn.Linear(dims[-2], 3),
        )
        self.pre_fc2 = nn.Sequential(
            nn.Linear(3, dims[-2]), nn.ReLU(), nn.Linear(dims[-2], dims[-2])
        )
        fc = DecoderGrp(in_dim=dims[-2], dim=dims[-2], down_ratio=1, knn_size=knn_sizes[-2])
        attn = [
            DecoderAttn(
                in_dim1=dims[-2],
                in_dim2=dims[-2],
                dim=attn_dim,
                pos_hid_dim=pos_hid_dim,
                attn_hid_dim=attn_hid_dim,
                knn_size=knn_sizes[-2],
                last=(i+1)==block_sizes[-2]
            )
            for i in range(block_sizes[-2])
        ]
        self.dec2 = nn.Sequential(fc, *attn)
        self.reg2 = RegionGrouping(dims[-2], n_regs, dims[-2], rand_reg_p)
        self.post_fc2 = nn.Sequential(
            nn.Linear(3 * dims[-2], dims[-2]),
            nn.ReLU(),
            nn.Linear(dims[-2], dims[-2]),
            nn.ReLU(),
            nn.Linear(dims[-2], 3),
        )

        self.grid1 = grid(up_ratio, 0.05).cuda().contiguous()
        self.g_pc11 = nn.Sequential(
            nn.Linear(dims[-1], dims[-1]),
            nn.ReLU(),
            nn.Linear(dims[-1], dims[-1]),
            nn.ReLU(),
        )
        self.g_pc12 = nn.Sequential(
            nn.Linear(dims[-1] + 2 + 3, dims[-1]),
            nn.ReLU(),
            nn.Linear(dims[-1], dims[-1]),
            nn.ReLU(),
            nn.Linear(dims[-1], 3),
        )
        self.pre_fc1 = nn.Sequential(
            nn.Linear(3, dims[-1]), nn.ReLU(), nn.Linear(dims[-1], dims[-1])
        )
        fc = DecoderGrp(in_dim=dims[-1], dim=dims[-1], down_ratio=1, knn_size=knn_sizes[-1])
        attn = [
            DecoderAttn(
                in_dim1=dims[-1],
                in_dim2=dims[-3],
                dim=attn_dim,
                pos_hid_dim=pos_hid_dim,
                attn_hid_dim=attn_hid_dim,
                knn_size=knn_sizes[-1],
                last=(i+1)==block_sizes[-1]
            )
            for i in range(block_sizes[-1])
        ]
        self.dec1 = nn.Sequential(fc, *attn)
        self.reg1 = RegionGrouping(dims[-1], n_regs, dims[-1], rand_reg_p)
        self.post_fc1 = nn.Sequential(
            nn.Linear(3 * dims[-1], dims[-1]),
            nn.ReLU(),
            nn.Linear(dims[-1], dims[-1]),
            nn.ReLU(),
            nn.Linear(dims[-1], 3),
        )

    def forward(self, g, enc):
        g1 = self.g_fc1(g)
        g2 = self.g_fc2(g1)
        g3 = self.g_fc3(g2)

        pc3 = self.g_pc3(g3)
        pc3 = pc3.view(pc3.shape[0], -1, 3)
        dec3 = self.pre_fc3(pc3)
        dec3 = self.dec3((dec3, pc3, *enc[-1]))[0]
        pc3 = self.post_fc3(dec3)

        coarse = pc3[:, :, None, :].repeat(1, 1, self.grid2.shape[0], 1).view(g2.shape[0], -1, 3)
        g2 = self.g_pc21(g2)
        grid2 = self.grid2[None, :, :].repeat(g2.shape[0], pc3.shape[1], 1)
        feat = torch.cat([grid2, coarse, g2[:, None, :].repeat(1, coarse.shape[1], 1)], dim=2)
        pc2 = self.g_pc22(feat) + coarse
        dec2 = self.pre_fc2(pc2)
        dec2 = self.dec2((dec2, pc2, *enc[-2]))[0]
        dec2, reg_loss1 = self.reg2(dec2, g2)
        pc2 = self.post_fc2(dec2) + pc2

        coarse = pc2[:, :, None, :].repeat(1, 1, self.grid1.shape[0], 1).view(g1.shape[0], -1, 3)
        g1 = self.g_pc11(g1)
        grid1 = self.grid1[None, :, :].repeat(g1.shape[0], pc2.shape[1], 1)
        feat = torch.cat([grid1, coarse, g1[:, None, :].repeat(1, coarse.shape[1], 1)], dim=2)
        pc1 = self.g_pc12(feat) + coarse
        dec1 = self.pre_fc1(pc1)
        dec1 = self.dec1((dec1, pc1, *enc[-3]))[0]
        dec1, reg_loss2, masks = self.reg1(dec1, g1, return_masks=True)
        pc1 = self.post_fc1(dec1) + pc1

        return pc3, pc2, pc1, (reg_loss1 + reg_loss2) / 2, masks


class ClassificationNetwork(nn.Module):
    def __init__(
        self, dims, knn_sizes, down_ratio, block_sizes, attn_dim, pos_hid_dim, attn_hid_dim, n_cls
    ):
        super(ClassificationNetwork, self).__init__()
        self.encoder = Encoder(
            dims=dims,
            knn_sizes=knn_sizes,
            down_ratio=down_ratio,
            block_sizes=block_sizes,
            attn_dim=attn_dim,
            pos_hid_dim=pos_hid_dim,
            attn_hid_dim=attn_hid_dim,
        )
        self.fc = nn.Linear(dims[-1], n_cls)

    def forward(self, x):
        x = self.encoder((x, x))
        x = x[-1][0].mean(dim=1)
        x = self.fc(x)
        return x


class CompletionNetwork(nn.Module):
    def __init__(
        self,
        dims,
        knn_sizes,
        down_ratio,
        block_sizes,
        attn_dim,
        pos_hid_dim,
        attn_hid_dim,
        n_pts,
        n_regs,
        rand_reg_p,
    ):
        super(CompletionNetwork, self).__init__()
        self.encoder = Encoder(
            dims=dims,
            knn_sizes=knn_sizes,
            down_ratio=down_ratio,
            block_sizes=block_sizes,
            attn_dim=attn_dim,
            pos_hid_dim=pos_hid_dim,
            attn_hid_dim=attn_hid_dim,
        )
        self.decoder = Decoder(
            dims=dims,
            n_pts=n_pts,
            up_ratio=down_ratio,
            knn_sizes=knn_sizes,
            block_sizes=block_sizes,
            attn_dim=attn_dim,
            pos_hid_dim=pos_hid_dim,
            attn_hid_dim=attn_hid_dim,
            n_regs=n_regs,
            rand_reg_p=rand_reg_p,
        )

    def forward(self, x):
        x = self.encoder((x, x))
        g = x[-1][0].mean(dim=1)
        x = self.decoder(g, x)
        return x


if __name__ == "__main__":
    # net = ClassificationNetwork(
    #     dims=[64, 128, 256],
    #     knn_sizes=[16, 16, 16],
    #     down_ratio=4,
    #     block_sizes=[2, 2, 2],
    #     attn_dim=64,
    #     pos_hid_dim=64,
    #     attn_hid_dim=256,
    #     n_cls=10,
    # ).cuda()
    # pc = torch.randn(2, 1024, 3).cuda()
    # o = net(pc)
    # print(o.shape)

    net = CompletionNetwork(
        dims=[64, 128, 256],
        knn_sizes=[16, 16, 16],
        down_ratio=4,
        block_sizes=[2, 2, 2],
        attn_dim=64,
        pos_hid_dim=64,
        attn_hid_dim=256,
        n_pts=1024,
        n_regs=4,
        rand_reg_p=0.1,
    ).cuda()
    pc = torch.randn(2, 1024, 3).cuda()
    p = net(pc)
    o, r = p[:-1], p[-1]
    print([o_.shape for o_ in o])
    print(r)
