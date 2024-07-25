import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torchvision import models
from .backbone import Encoder, Decoder, Bottleneck

from .RFBmodule import *

CHANNEL_EXPAND = {
    'resnet18': 1,
    'resnet34': 1,
    'resnet50': 4,
    'resnet101': 4
}


def Soft_aggregation(ps, max_obj):
    num_objects, H, W = ps.shape
    em = torch.zeros(1, max_obj + 1, H, W).to(ps.device)
    em[0, 0, :, :] = torch.prod(1 - ps, dim=0)  # bg prob
    em[0, 1:num_objects + 1, :, :] = ps  # obj prob
    em = torch.clamp(em, 1e-7, 1 - 1e-7)
    logit = torch.log((em / (1 - em)))

    return logit


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Encoder_M(nn.Module):
    def __init__(self, arch):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_bg = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1
        self.res3 = resnet.layer2
        self.res4 = resnet.layer3

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_m, in_bg):
        f = in_f
        m = torch.unsqueeze(in_m, dim=1).float()
        bg = torch.unsqueeze(in_bg, dim=1).float()

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_bg(bg)
        x = self.bn1(x)
        c1 = self.relu(x)
        x = self.maxpool(c1)
        r2 = self.res2(x)
        r3 = self.res3(r2)
        r4 = self.res4(r3)

        return r4, r3, r2, c1


class Encoder_Q(nn.Module):
    def __init__(self, arch):
        super(Encoder_Q, self).__init__()

        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1
        self.res3 = resnet.layer2
        self.res4 = resnet.layer3

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f):
        f = in_f

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)
        x = self.maxpool(c1)
        r2 = self.res2(x)
        r3 = self.res3(r2)
        r4 = self.res4(r3)

        return r4, r3, r2, c1


class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:], mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, inplane, mdim, expand):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(inplane, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(128 * expand, mdim)
        self.RF2 = Refine(64 * expand, mdim)

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2, f):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)
        m2 = self.RF2(r2, m3)

        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2, size=f.shape[2:], mode='bilinear', align_corners=False)
        return p


class Temporal_Memory(nn.Module):
    def __init__(self, feat_size=10, decay=0.6):
        super(Temporal_Memory, self).__init__()
        self.feat_size = feat_size
        self.decay = decay

    def forward(self, m_in, m_out, q_in, q_out):
        no, _, H, W = q_in.size()
        centers, C = m_in.size()
        _, vd = m_out.shape

        qi = q_in.reshape(-1, C, H * W)
        p = torch.matmul(qi.permute(0, 2, 1), m_in.permute(1, 0))
        p = p / math.sqrt(C)
        if p.shape[0] > 1:
            for decay_idx in range(1, p.shape[0]):
                p[decay_idx, :, :] = self.decay * p[decay_idx, :, :]
        p = torch.softmax(p, dim=1)

        mem = torch.matmul(p, m_out)
        mem = mem.permute(0, 2, 1).reshape(no, vd, H, W)
        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p


class Spatial_Memory(nn.Module):
    def __init__(self, feat_size=15, decay=0.92):
        super(Spatial_Memory, self).__init__()
        self.feat_size = feat_size
        self.decay = decay
        x, y = torch.meshgrid(torch.arange(self.feat_size), torch.arange(self.feat_size))
        coordinates = torch.stack((x, y), dim=-1).view(-1, 2)
        x = torch.abs(coordinates[:, 0].unsqueeze(1) - coordinates[:, 0].unsqueeze(0))  # x.numpy()
        y = torch.abs(coordinates[:, 1].unsqueeze(1) - coordinates[:, 1].unsqueeze(0))  # y.numpy()
        self.weight_decay_mask = ((self.decay ** x) * (self.decay ** y))

    def forward(self, m_in, m_out, q_in, q_out):
        B, _, H, W = q_in.size()
        no, centers, C = m_in.size()
        _, _, vd = m_out.shape
        qi = q_in.view(-1, C, H * W)
        p = torch.bmm(m_in, qi)
        p = p / math.sqrt(C)
        p = p * self.weight_decay_mask.to(p.device).repeat(B, 1, 1)
        p = torch.softmax(p, dim=1)
        mo = m_out.permute(0, 2, 1)
        mem = torch.bmm(mo, p)
        mem = mem.view(no, vd, H, W)
        mem_out = torch.cat([mem, q_out], dim=1)
        return mem_out, p


class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class Conv_decouple(nn.Module):

    def __init__(self, inplanes, planes):
        super(Conv_decouple, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class STAN(nn.Module):

    def __init__(self, opt):
        super(STAN, self).__init__()

        keydim = opt.keydim
        valdim = opt.valdim
        arch = opt.arch
        expand = CHANNEL_EXPAND[arch]
        self.Encoder_M = Encoder_M(arch)
        self.Encoder_Q = Encoder_Q(arch)
        self.keydim = keydim
        self.valdim = valdim
        self.KV_M_r4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim)
        self.KV_Q_r4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim)
        self.KV_m4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim)
        self.rfb_key = BasicRFB(2 * opt.sampled_frames, opt.sampled_frames)
        self.rfb_val = BasicRFB(2 * opt.sampled_frames, opt.sampled_frames)
        self.Memory1 = Temporal_Memory(decay=opt.temporal_decay)
        self.SpatialMemory = Spatial_Memory(decay=opt.spatial_decay)
        self.conv_decouple = Conv_decouple(2048, 1024)
        self.Decoder = Decoder(2 * valdim, 256, expand)

    def load_param(self, weight):

        s = self.state_dict()
        for key, val in weight.items():
            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))

        self.load_state_dict(s)

    def memorize(self, frame, masks, num_objects):
        frame_batch = []
        mask_batch = []
        bg_batch = []
        for o in range(1, num_objects + 1):
            frame_batch.append(frame)
            mask_batch.append(masks[:, o])

        for o in range(1, num_objects + 1):
            bg_batch.append(torch.clamp(1.0 - masks[:, o], min=0.0, max=1.0))

        # make Batch
        frame_batch = torch.cat(frame_batch, dim=0)
        mask_batch = torch.cat(mask_batch, dim=0)
        bg_batch = torch.cat(bg_batch, dim=0)

        r4, _, _, _ = self.Encoder_M(frame_batch, mask_batch, bg_batch)
        _, c, h, w = r4.size()
        k4, v4 = self.KV_M_r4(r4)
        k4, v4 = k4.reshape(k4.size(0), k4.size(1), -1).permute(0, 2, 1), v4.reshape(v4.size(0), v4.size(1),
                                                                                     -1).permute(0, 2, 1)
        return k4, v4, r4

    def segment(self, frame, keys, values, num_objects, max_obj, opt, Clip_idx, keys_dict, vals_dict, patch=2):
        r4, r3e, r2e, _ = self.Encoder_Q(frame)
        n, c, h, w = r4.size()
        k4e, v4e = self.KV_Q_r4(r4)  # 1, dim, H/16, W/16

        if opt.adapt_memory and (Clip_idx > opt.memory_max_Clip) and frame.size(0) <= opt.sampled_frames:
            adapt_keys, adapt_vals, score_dict = [], [], {}

            curr_key = k4e.reshape(k4e.shape[0], k4e.shape[1], -1).permute(0, 2, 1)
            for idx in range(1, len(keys_dict) + 1):
                score_dict[idx] = torch.cosine_similarity(curr_key, keys_dict[idx], dim=1).mean(dim=1).mean()

            Top_Wanted_idx_list = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_idx_list = [idx for idx, _ in Top_Wanted_idx_list]

            if Clip_idx != sorted_idx_list[-1] and Clip_idx != sorted_idx_list[-2]:
                keyclip1, keyclip2 = keys_dict[sorted_idx_list[-1]], keys_dict[sorted_idx_list[-2]]
                keycat = torch.cat((keyclip1, keyclip2), dim=0).unsqueeze(0).permute(0, 1, 3, 2)
                keycat = keycat.reshape(int(keycat.size(0)), int(keycat.size(1)), int(keycat.size(2)),
                                          int(math.sqrt(keycat.size(3))), int(math.sqrt(keycat.size(3))))
                conv_keycat = self.rfb_key(keycat).squeeze(0)
                conv_keycat = conv_keycat.reshape(int(conv_keycat.size(0)), int(conv_keycat.size(1)), -1).permute(0, 2, 1)

                valueclip1, valueclip2 = vals_dict[sorted_idx_list[-1]], vals_dict[sorted_idx_list[-2]]
                valcat = torch.cat((valueclip1, valueclip2), dim=0).unsqueeze(0).permute(0, 1, 3, 2)
                valcat = valcat.reshape(int(valcat.size(0)), int(valcat.size(1)), int(valcat.size(2)),
                                          int(math.sqrt(valcat.size(3))), int(math.sqrt(valcat.size(3))))
                conv_valclip = self.rfb_val(valcat).squeeze(0)
                conv_valclip = conv_valclip.reshape(int(conv_valclip.size(0)), int(conv_valclip.size(1)), -1).permute(0, 2, 1)

                adapt_keys.append(conv_keycat), adapt_vals.append(conv_valclip)

                sorted_idx_list = sorted_idx_list[: opt.memory_max_Clip-1]
                sorted_idx_list.sort()
                for idx in sorted_idx_list:
                    adapt_keys.append(keys_dict[idx])
                    adapt_vals.append(vals_dict[idx])

            else:
                sorted_idx_list = sorted_idx_list[: opt.memory_max_Clip]
                sorted_idx_list.sort()
                for idx in sorted_idx_list:
                    adapt_keys.append(keys_dict[idx]), adapt_vals.append(vals_dict[idx])

            keys = torch.cat(adapt_keys, dim=0)
            values = torch.cat(adapt_vals, dim=0)

            adapt_keys.clear(), adapt_vals.clear()
        m4 = torch.zeros_like(r4)
        BT_ks, HW_ks, C_ks = keys.size()
        keys = keys.reshape(BT_ks, int(math.sqrt(HW_ks)), int(math.sqrt(HW_ks)), C_ks)
        BT_ks, H_ks, W_ks, C_ks = keys.size()

        BT_vs, HW_vs, C_vs = values.size()
        values = values.reshape(BT_vs, int(math.sqrt(HW_vs)), int(math.sqrt(HW_vs)), C_vs)
        BT_vs, H_vs, W_vs, C_vs = values.size()
        assert H_ks == H_vs and W_ks == W_vs
        cut_H, cut_W = H_ks // patch, W_ks // patch
        gap_H, gap_W = int(cut_H // 2), int(cut_W // 2)
        keys_patch, values_patch = keys[:, 0:(cut_H+gap_H), 0:(cut_W+gap_W), :], values[:, 0:(cut_H+gap_H), 0:(cut_W+gap_W), :]
        keys_patch = keys_patch.reshape(keys_patch.size(0), (cut_H+gap_H) * (cut_W+gap_W), keys_patch.size(3))
        keys_patch = keys_patch.reshape(keys_patch.size(0) * (cut_H+gap_H) * (cut_W+gap_W), keys_patch.size(2))
        values_patch = values_patch.reshape(values_patch.size(0), (cut_H+gap_H) * (cut_W+gap_W), values_patch.size(3))
        values_patch = values_patch.reshape(values_patch.size(0) * (cut_H+gap_H) * (cut_W+gap_W), values_patch.size(2))
        k4e_patch, v4e_patch = k4e[:(k4e.size(0) - 1), :, 0:(cut_H + gap_H), 0:(cut_W + gap_W)], v4e[:(v4e.size(0) - 1),
                                                                                                 :, 0:(cut_H + gap_H),
                                                                                                 0:(cut_W + gap_W)]
        m4_patch_first, _ = self.Memory1(keys_patch, values_patch, k4e_patch, v4e_patch)
        keys_patch = torch.cat([keys_patch, k4e_patch.permute(1, 0, 2, 3).reshape(k4e_patch.size(1), -1).permute(1, 0)], dim=0)
        values_patch = torch.cat([values_patch, v4e_patch.permute(1, 0, 2, 3).reshape(v4e_patch.size(1), -1).permute(1, 0)], dim=0)
        k4e_patch, v4e_patch = k4e[-1, :, 0:(cut_H + gap_H), 0:(cut_W + gap_W)].unsqueeze(0), v4e[-1, :, 0:(cut_H + gap_H), 0:(cut_W + gap_W)].unsqueeze(0)
        m4_patch_last, _ = self.Memory1(keys_patch, values_patch, k4e_patch, v4e_patch)
        m4_patch = torch.cat([m4_patch_first, m4_patch_last], dim=0)
        m4[:, :, 0:cut_H, 0:cut_W] = m4_patch[:, :, 0:cut_H, 0:cut_W]
        keys_patch, values_patch = keys[:, 0:cut_H, cut_W:, :], values[:, 0:cut_H, cut_W:, :]
        keys_patch = keys_patch.reshape(keys_patch.size(0), cut_H * (W_ks - cut_W), keys_patch.size(3))
        keys_patch = keys_patch.reshape(keys_patch.size(0) * cut_H * (W_ks - cut_W), keys_patch.size(2))
        values_patch = values_patch.reshape(values_patch.size(0), cut_H * (W_vs - cut_W),
                                            values_patch.size(3))
        values_patch = values_patch.reshape(values_patch.size(0) * cut_H * (W_vs - cut_W),
                                            values_patch.size(2))
        k4e_patch, v4e_patch = k4e[:(k4e.size(0) - 1), :, 0:cut_H, cut_W:], v4e[:(v4e.size(0) - 1), :, 0:cut_H, cut_W:]
        m4_patch_first, _ = self.Memory1(keys_patch, values_patch, k4e_patch, v4e_patch)
        keys_patch = torch.cat([keys_patch, k4e_patch.permute(1, 0, 2, 3).reshape(k4e_patch.size(1), -1).permute(1, 0)],
                               dim=0)
        values_patch = torch.cat(
            [values_patch, v4e_patch.permute(1, 0, 2, 3).reshape(v4e_patch.size(1), -1).permute(1, 0)], dim=0)
        k4e_patch, v4e_patch = k4e[-1, :, 0:cut_H, cut_W:].unsqueeze(0), v4e[-1, :, 0:cut_H, cut_W:].unsqueeze(0)
        m4_patch_last, _ = self.Memory1(keys_patch, values_patch, k4e_patch, v4e_patch)

        m4_patch = torch.cat([m4_patch_first, m4_patch_last], dim=0)
        m4[:, :, 0:cut_H, cut_W:] = m4_patch
        keys_patch, values_patch = keys[:, cut_H:, 0:cut_W, :], values[:, cut_H:, 0:cut_W, :]
        keys_patch = keys_patch.reshape(keys_patch.size(0), (H_ks - cut_H) * cut_W, keys_patch.size(3))
        keys_patch = keys_patch.reshape(keys_patch.size(0) * (H_ks - cut_H) * cut_W, keys_patch.size(2))
        values_patch = values_patch.reshape(values_patch.size(0), (H_vs - cut_H) * cut_W,
                                            values_patch.size(3))
        values_patch = values_patch.reshape(values_patch.size(0) * (H_vs - cut_H) * cut_W,
                                            values_patch.size(2))
        k4e_patch, v4e_patch = k4e[:(k4e.size(0) - 1), :, cut_H:, 0:cut_W], v4e[:(v4e.size(0) - 1), :, cut_H:, 0:cut_W]
        m4_patch_first, _ = self.Memory1(keys_patch, values_patch, k4e_patch, v4e_patch)
        keys_patch = torch.cat([keys_patch, k4e_patch.permute(1, 0, 2, 3).reshape(k4e_patch.size(1), -1).permute(1, 0)],
                               dim=0)
        values_patch = torch.cat(
            [values_patch, v4e_patch.permute(1, 0, 2, 3).reshape(v4e_patch.size(1), -1).permute(1, 0)], dim=0)
        k4e_patch, v4e_patch = k4e[-1, :, cut_H:, 0:cut_W].unsqueeze(0), v4e[-1, :, cut_H:, 0:cut_W].unsqueeze(0)
        m4_patch_last, _ = self.Memory1(keys_patch, values_patch, k4e_patch, v4e_patch)
        m4_patch = torch.cat([m4_patch_first, m4_patch_last], dim=0)
        m4[:, :, cut_H:, 0:cut_W] = m4_patch
        keys_patch, values_patch = keys[:, cut_H:, cut_W:, :], values[:, cut_H:, cut_W:, :]
        keys_patch = keys_patch.reshape(keys_patch.size(0), (H_ks - cut_H) * (W_ks - cut_W),
                                        keys_patch.size(3))
        keys_patch = keys_patch.reshape(keys_patch.size(0) * (H_ks - cut_H) * (W_ks - cut_W),
                                        keys_patch.size(2))
        values_patch = values_patch.reshape(values_patch.size(0), (H_vs - cut_H) * (W_vs - cut_W),
                                            values_patch.size(3))
        values_patch = values_patch.reshape(values_patch.size(0) * (H_vs - cut_H) * (W_vs - cut_W),
                                            values_patch.size(2))
        k4e_patch, v4e_patch = k4e[:(k4e.size(0) - 1), :, cut_H:, cut_W:], v4e[:(v4e.size(0) - 1), :, cut_H:, cut_W:]
        m4_patch_first, _ = self.Memory1(keys_patch, values_patch, k4e_patch, v4e_patch)
        keys_patch = torch.cat([keys_patch, k4e_patch.permute(1, 0, 2, 3).reshape(k4e_patch.size(1), -1).permute(1, 0)],
                               dim=0)
        values_patch = torch.cat(
            [values_patch, v4e_patch.permute(1, 0, 2, 3).reshape(v4e_patch.size(1), -1).permute(1, 0)], dim=0)
        k4e_patch, v4e_patch = k4e[-1, :, cut_H:, cut_W:].unsqueeze(0), v4e[-1, :, cut_H:, cut_W:].unsqueeze(0)
        m4_patch_last, _ = self.Memory1(keys_patch, values_patch, k4e_patch, v4e_patch)

        m4_patch = torch.cat([m4_patch_first, m4_patch_last], dim=0)
        m4[:, :, cut_H:, cut_W:] = m4_patch
        m4_spatial_list = []
        km4, vm4 = self.KV_m4(r4)
        for c in range(r4.size(0)):
            if c == 0:
                memo_keys = keys[-1, :, :, :].unsqueeze(0)
                memo_keys = memo_keys.reshape(memo_keys.size(0), -1, memo_keys.size(3))
                memo_values = values[-1, :, :, :].unsqueeze(0)
                memo_values = memo_values.reshape(memo_values.size(0), -1, memo_values.size(3))
                frame_spatial, _ = self.SpatialMemory(memo_keys, memo_values, km4[c, :, :, :].unsqueeze(0),
                                                      vm4[c, :, :, :].unsqueeze(0))
            else:
                memo_keys = km4[c - 1, :, :, :].unsqueeze(0)
                memo_keys = memo_keys.reshape(memo_keys.size(0), memo_keys.size(1), -1).permute(0, 2, 1)
                memo_values = vm4[c - 1, :, :, :].unsqueeze(0)
                memo_values = memo_values.reshape(memo_values.size(0), memo_values.size(1), -1).permute(0, 2, 1)
                frame_spatial, _ = self.SpatialMemory(memo_keys, memo_values, km4[c, :, :, :].unsqueeze(0),
                                                      vm4[c, :, :, :].unsqueeze(0))
            m4_spatial_list.append(frame_spatial)
        m4_spatial = torch.cat(m4_spatial_list, dim=0)
        m4 = torch.cat((m4, m4_spatial), dim=1)
        m4 = self.conv_decouple(m4)
        logit = self.Decoder(m4, r3e, r2e, frame)
        ps = F.softmax(logit, dim=1)[:, 1]
        logit_list = []
        for f in range(ps.size(0)):
            logit_list.append(Soft_aggregation(ps[f, :, :].unsqueeze(0), max_obj))

        return logit_list, ps

    def forward(self, frame, mask=None, keys=None, values=None, num_objects=None, max_obj=None,
                opt=None, Clip_idx=None, keys_dict=None, vals_dict=None, patch=2):

        if mask is not None:
            return self.memorize(frame, mask, num_objects)
        else:
            return self.segment(frame, keys, values, num_objects, max_obj, opt, Clip_idx, keys_dict, vals_dict, patch)