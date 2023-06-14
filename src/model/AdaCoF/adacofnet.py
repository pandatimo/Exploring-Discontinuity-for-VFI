import torch
import model.AdaCoF.cupy_module.adacof as adacof
import sys
from torch.nn import functional as F


class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
        )

        def Subnet_offset(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        def Subnet_weight(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        def Subnet_occlusion():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                torch.nn.Sigmoid()
            )

        def Subnet_Dmap():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                torch.nn.Sigmoid()
            )

        self.moduleConv1 = Basic(12, 64)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(256, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = Upsample(512)

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = Upsample(256)

        self.moduleDeconv3 = Basic(256, 256)
        self.moduleUpsample3 = Upsample(256)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128)

        self.moduleWeight1 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleWeight2 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha2 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta2 = Subnet_offset(self.kernel_size ** 2)
        self.moduleOcclusion = Subnet_occlusion()
        self.moduleD = Subnet_Dmap()


    def forward(self, rfield):
        tensorConv1 = self.moduleConv1(rfield)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2
        tensorCombine_ = tensorCombine[:, 32:96, :, :]
        Weight1 = self.moduleWeight1(tensorCombine_)
        Alpha1 = self.moduleAlpha1(tensorCombine_)
        Beta1 = self.moduleBeta1(tensorCombine_)
        Weight2 = self.moduleWeight2(tensorCombine_)
        Alpha2 = self.moduleAlpha2(tensorCombine_)
        Beta2 = self.moduleBeta2(tensorCombine_)
        Occlusion = self.moduleOcclusion(tensorCombine_)
        Dmap = self.moduleD(tensorCombine)

        return Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion, Dmap


class AdaCoFNet(torch.nn.Module):
    def __init__(self):
        super(AdaCoFNet, self).__init__()
        self.kernel_size = 5
        self.kernel_pad = int(((5 - 1) * 1) / 2.0)
        self.dilation = 1

        self.get_kernel = KernelEstimation(self.kernel_size)

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

    def forward(self, frames):
        frame1 = frames[0]
        frame2 = frames[1]
        frame3 = frames[2]
        frame4 = frames[3]
        h0 = int(list(frame1.size())[2])
        w0 = int(list(frame1.size())[3])

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame1 = F.pad(frame1, (0, 0, 0, pad_h), mode='reflect')
            frame2 = F.pad(frame2, (0, 0, 0, pad_h), mode='reflect')
            frame3 = F.pad(frame3, (0, 0, 0, pad_h), mode='reflect')
            frame4 = F.pad(frame4, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame1 = F.pad(frame1, (0, pad_w, 0, 0), mode='reflect')
            frame2 = F.pad(frame2, (0, pad_w, 0, 0), mode='reflect')
            frame3 = F.pad(frame3, (0, pad_w, 0, 0), mode='reflect')
            frame4 = F.pad(frame4, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True

        Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Occlusion, Dmap = self.get_kernel(torch.cat((frame1, frame2, frame3, frame4), 1))

        tensorAdaCoF1 = self.moduleAdaCoF(self.modulePad(frame2), Weight1, Alpha1, Beta1, self.dilation)
        tensorAdaCoF2 = self.moduleAdaCoF(self.modulePad(frame3), Weight2, Alpha2, Beta2, self.dilation)

        i_frame = Occlusion * tensorAdaCoF1 + (1 - Occlusion) * tensorAdaCoF2

        i_frame = frame2 * Dmap + i_frame * (1 - Dmap)

        if h_padded:
            i_frame = i_frame[:, :, 0:h0, :]
            Dmap = Dmap[:, :, 0:h0, :]
        if w_padded:
            i_frame = i_frame[:, :, :, 0:w0]
            Dmap = Dmap[:, :, :, 0:w0]

        return i_frame
