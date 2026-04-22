import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from thop import profile
import time

class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, 3))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm1d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, L = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        channel_max = x.view(N, C, -1).max(dim=2, keepdim=True)[0]

        t = torch.cat((channel_mean, channel_std,channel_max), dim=2)
        return t

    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)  # B x C

        z_hat = self.bn(z)
        g = self.activation(z_hat)
        g = g[:, :, None]  # B x C x 1

        return g

    def forward(self, x):
        t = self._style_pooling(x)
        g = self._style_integration(t)

        return x * g


class Conv1dDownMptBlock(nn.Module):
    def __init__(self, nin=8, nout=11, ks=7, st=4, padding=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(nin, nin, ks, st, padding=padding, groups=nin),  # Depthwise convolution
            nn.ReLU(),
            nn.Conv1d(nin, nout, 1),  # Pointwise convolution
            nn.ReLU(),
            SRMLayer(nout)  # SRM layer
        )

    def forward(self, x):
        x = self.layers(x)
        return x
class Conv1dUpMptBlock(nn.Module):
    def __init__(self, nin=8, nout=11, ks=7, st=4, padding=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(nin * 2, nin * 2, ks, 1, padding=padding),  # 普通卷积
            nn.ReLU(),
            nn.ConvTranspose1d(nin * 2, nout, ks if nin not in [16, 11] else ks + 1, st, padding=padding - 1),
            nn.ReLU()
        )

    # def forward(self, x, skip_x):
    #     x = torch.cat([x, skip_x], dim=1)
    #     x = self.layers(x)
    #     return x



    def forward(self, x, skip_x):
        # Adjust skip_x size to match x
        if x.size(-1) != skip_x.size(-1):
            diff = skip_x.size(-1) - x.size(-1)
            if diff % 2 == 0:
                skip_x = skip_x[..., diff // 2: -(diff // 2)]
            else:
                skip_x = skip_x[..., diff // 2: -(diff // 2) - 1]
        x = torch.cat([x, skip_x], dim=1)
        return self.layers(x)








# class Conv1dUpMptBlock(nn.Module):
#     def __init__(self, nin=8, nout=11, ks=7, st=4, padding=3):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Conv1d(nin * 2, nin * 2, ks, 1, padding=padding, groups=nin * 2),  # Depthwise convolution
#             nn.ReLU(),
#             nn.ConvTranspose1d(nin * 2, nout, ks if nin not in [16, 11] else ks + 1, st, padding=padding - 1),
#             nn.ReLU(),
#             SRMLayer(nout)  # SRM layer
#         )
#
#     def forward(self, x, skip_x):
#         x = torch.cat([x, skip_x], dim=1)
#         x = self.layers(x)
#         return x


class UNet_mpt(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(3, 8, 7, 1, padding=3),
            nn.ReLU()
        )

        self.down_layer1 = Conv1dDownMptBlock(8, 11, 7, 4, padding=3)
        self.down_layer2 = Conv1dDownMptBlock(11, 16, 7, 4, padding=3)
        self.down_layer3 = Conv1dDownMptBlock(16, 22, 7, 4, padding=3)
        self.down_layer4 = Conv1dDownMptBlock(22, 32, 7, 4, padding=3)

        self.up_sample = nn.Sequential(
            nn.ConvTranspose1d(32, 22, 6, 4, padding=2),
            nn.ReLU()
        )

        self.up_layer3 = Conv1dUpMptBlock(22, 16, 7, 4, padding=3)
        self.up_layer2 = Conv1dUpMptBlock(16, 11, 7, 4, padding=3)
        self.up_layer1 = Conv1dUpMptBlock(11, 8, 7, 4, padding=3)

        self.output = nn.Sequential(
            nn.Conv1d(16, 3, 7, 1, padding=3),
            nn.ReLU(),
            nn.Conv1d(3, 1, 7, 1, padding=3)
        )

    # def forward(self, x):
    #     x = self.input(x)
    #     down_x1 = self.down_layer1(x)
    #     down_x2 = self.down_layer2(down_x1)
    #     down_x3 = self.down_layer3(down_x2)
    #     down_x4 = self.down_layer4(down_x3)
    #     up_x4 = self.up_sample(down_x4)
    #     up_x3 = self.up_layer3(up_x4, down_x3)
    #     up_x2 = self.up_layer2(up_x3, down_x2)
    #     up_x1 = self.up_layer1(up_x2, down_x1)
    #     # print('down_x1: ', down_x1.shape)
    #     output = self.output(torch.cat([up_x1, x], dim=1))
    #     print(f"up_x2 shape: {up_x2.shape}, down_x2 shape: {down_x2.shape}")
    #
    #     return output
    #

    def forward(self, x):
        x = self.input(x)
        down_x1 = self.down_layer1(x)
        down_x2 = self.down_layer2(down_x1)
        down_x3 = self.down_layer3(down_x2)
        down_x4 = self.down_layer4(down_x3)
        up_x4 = self.up_sample(down_x4)
        if up_x4.size(-1) != down_x3.size(-1):
            diff = down_x3.size(-1) - up_x4.size(-1)
            up_x4 = nn.functional.pad(up_x4, (diff // 2, diff - diff // 2))

        up_x3 = self.up_layer3(up_x4, down_x3)
        if up_x3.size(-1) != down_x2.size(-1):
            diff = down_x2.size(-1) - up_x3.size(-1)
            up_x3 = nn.functional.pad(up_x3, (diff // 2, diff - diff // 2))

        up_x2 = self.up_layer2(up_x3, down_x2)
        if up_x2.size(-1) != down_x1.size(-1):
            diff = down_x1.size(-1) - up_x2.size(-1)
            up_x2 = nn.functional.pad(up_x2, (diff // 2, diff - diff // 2))

        up_x1 = self.up_layer1(up_x2, down_x1)
        if up_x1.size(-1) != x.size(-1):
            diff = x.size(-1) - up_x1.size(-1)
            up_x1 = nn.functional.pad(up_x1, (diff // 2, diff - diff // 2))

        output = self.output(torch.cat([up_x1, x], dim=1))
        return output






class Conv1dDownBlock(nn.Module):
    def __init__(self, nin=8, nout=11, ks=7, st=4, padding=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(nin, nin, ks, st, padding=padding, groups=nin),  # 深度可分离卷积
            nn.ReLU(),
            nn.Conv1d(nin, nout, 1),  # 逐点卷积
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class Conv1dUpBlock(nn.Module):
    def __init__(self, nin=8, nout=11, ks=7, st=4, padding=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(nin * 2, nin * 2, ks, 1, padding=padding, groups=nin * 2),  # 深度可分离卷积
            nn.ReLU(),
            nn.ConvTranspose1d(nin * 2, nout, ks if nin not in [16, 11] else ks + 1, st, padding=padding-1),
            nn.ReLU()
        )

    def forward(self, x, skip_x):
        x = torch.cat([x, skip_x], dim=1)
        x = self.layers(x)
        return x

class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(
                nn.Conv1d(3, 8, 7, 1, padding=3),
                nn.ReLU()
            )

            self.down_layer1 = Conv1dDownBlock(8, 11, 7, 4, padding=3)
            self.down_layer2 = Conv1dDownBlock(11, 16, 7, 4, padding=3)
            self.down_layer3 = Conv1dDownBlock(16, 22, 7, 4, padding=3)
            self.down_layer4 = Conv1dDownBlock(22, 32, 7, 4, padding=3)

            self.up_sample = nn.Sequential(
                nn.ConvTranspose1d(32, 22, 6, 4, padding=2),
                nn.ReLU()
            )

            self.up_layer3 = Conv1dUpBlock(22, 16, 7, 4, padding=3)
            self.up_layer2 = Conv1dUpBlock(16, 11, 7, 4, padding=3)
            self.up_layer1 = Conv1dUpBlock(11, 8, 7, 4, padding=3)

            self.output = nn.Sequential(
                nn.Conv1d(16, 3, 7, 1, padding=3),
                nn.ReLU(),
                nn.Conv1d(3, 1, 7, 1, padding=3)
            )
        #
        # def forward(self, x):
        #     x = self.input(x)
        #     down_x1 = self.down_layer1(x)
        #     down_x2 = self.down_layer2(down_x1)
        #     down_x3 = self.down_layer3(down_x2)
        #     down_x4 = self.down_layer4(down_x3)
        #     up_x4 = self.up_sample(down_x4)
        #     up_x3 = self.up_layer3(up_x4, down_x3)
        #     up_x2 = self.up_layer2(up_x3, down_x2)
        #     up_x1 = self.up_layer1(up_x2, down_x1)
        #     output = self.output(torch.cat([up_x1, x], dim=1))
        #
        #     print("up_x4 size:", up_x4.size())
        #     print("down_x3 size:", down_x3.size())
        #
        #     return output
        def forward(self, x):
            x = self.input(x)
            # print(f'Input size: {x.size()}')

            down_x1 = self.down_layer1(x)
            # print(f'down_x1 size: {down_x1.size()}')

            down_x2 = self.down_layer2(down_x1)
            # print(f'down_x2 size: {down_x2.size()}')

            down_x3 = self.down_layer3(down_x2)
            # print(f'down_x3 size: {down_x3.size()}')

            down_x4 = self.down_layer4(down_x3)
            # print(f'down_x4 size: {down_x4.size()}')

            up_x4 = self.up_sample(down_x4)
            # print(f'up_x4 size: {up_x4.size()}')

            up_x3 = self.up_layer3(up_x4, down_x3)
            # print(f'up_x3 size: {up_x3.size()}')

            up_x2 = self.up_layer2(up_x3, down_x2)
            # print(f'up_x2 size: {up_x2.size()}')

            up_x1 = self.up_layer1(up_x2, down_x1)
            # print(f'up_x1 size: {up_x1.size()}')

            output = self.output(torch.cat([up_x1, x], dim=1))
            # print(f'Output size: {output.size()}')

            return output



if __name__ == "__main__":
    model = UNet_mpt()
    # model=UNet()
    x = torch.randn([10, 3, 6000])#staford
    # x = torch.randn([4, 3, 12000])#instance
    #

    y = model(x)
    print(y.shape)

