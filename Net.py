import torch
import torch.nn as nn
import torch.nn.functional as F
import piq

class AIOLayer(nn.Module):
    def __init__(self, mc, hc):
        super(AIOLayer, self).__init__()
        # mc: muti-channel, hc: hyperspectral channel
        self.mc = mc
        self.hc = hc
        lin = mc + hc
        self.lin = lin

        self.mlp = nn.Conv2d(lin, lin, 1, 1, 0, bias=True)
        self.mmlp = nn.Conv2d(lin, lin, 1, 1, 0, bias=True)
        self.hmlp = nn.Conv2d(lin, lin, 1, 1, 0, bias=True)

        self.flag = 'train'

        self.reset_parameters()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def set_mhmlp_w(self):
        weightm = self.mmlp.weight.data.squeeze(-1).squeeze(-1)
        weightm[self.mc:,:] = 0
        weightm[:,self.mc:] = 0
        self.mmlp.weight.data = weightm.unsqueeze(-1).unsqueeze(-1)

        weighth = self.hmlp.weight.data.squeeze(-1).squeeze(-1)
        weighth[:self.mc,:] = 0
        weighth[:,:self.mc] = 0
        self.hmlp.weight.data = weighth.unsqueeze(-1).unsqueeze(-1)

    def fuse_mlp_bn(self, fc, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = bn.weight / std
        t = t.reshape(-1, 1, 1, 1)
        return fc.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def convert_mlps(self):
        self.set_mhmlp_w()
        fw,mw,hw = self.mlp.weight, self.mmlp.weight, self.hmlp.weight
        fb,mb,hb = self.mlp.bias, self.mmlp.bias, self.hmlp.bias
        w,b = fw + mw + hw, fb + mb + hb
        self.aiol = nn.Conv2d(self.lin, self.lin, 1, 1, 0)
        self.aiol.weight.data = w
        self.aiol.bias.data = b
        self.__delattr__('mlp')
        self.__delattr__('mmlp')
        self.__delattr__('hmlp')
        self.flag='eval'

    def forward(self, x):
        if self.flag == 'train':
            self.set_mhmlp_w()
            mlp_res =  self.mlp(x)
            mmlp_res = self.mmlp(x)
            hmlp_res = self.hmlp(x)
            res = mlp_res + mmlp_res + hmlp_res
        else:
            res = self.aiol(x) 
        return res

class SE(nn.Module):
    def __init__(self, inchannel):
        super(SE, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Sequential(nn.Conv2d(inchannel, inchannel//4, 1, 1, 0, bias=False),
                                     nn.PReLU())
        self.expand = nn.Conv2d(inchannel//4, inchannel, 1, 1, 0, bias=False)

    def forward(self, x):
        gap = self.gap(x)
        weight = torch.sigmoid(self.expand(self.squeeze(gap)))
        x = x*weight
        return x

class AIOBlock(nn.Module):
    def __init__(self, inc, inc2, ouc):
        super(AIOBlock, self).__init__()
        self.aol = AIOLayer(inc, inc2)
        self.d = nn.Sequential(nn.Conv2d(inc+inc2, ouc, 1, 1, 1//2, bias=True, groups=1),
                                nn.PReLU(),
                               )

        self.d2 = nn.Sequential(nn.Conv2d(ouc, ouc, 7, 1, 7//2, bias=True, groups=ouc),
                               )


        self.se = SE(ouc)

    def forward(self, x):
        d1 = self.d(self.aol(x))
        d1 = self.se(d1)
        return self.d2(d1) + d1

class Net(nn.Module):
    def __init__(self, mc, hc, order=4):
        super(Net, self).__init__()
        self.order = order
        self.pus = nn.PixelUnshuffle(order)
        self.ps = nn.PixelShuffle(order)
        channel = 32
        mcc = mc*order**2
        self.ao1 = AIOBlock(mcc, hc, channel)
        self.ao2 = AIOBlock(mcc, channel, channel*2)
        self.ao3 = AIOBlock(mcc, channel*2, channel*4)
        self.ao4 = AIOBlock(mcc, channel*4, channel*8)
        self.ao5 = AIOBlock(mcc, channel*8, channel*16)

        self.conv = nn.Sequential(nn.Conv2d(channel*16, channel*16, 1, 1, 0, bias=True),
                                  nn.PReLU(),
                                  nn.Conv2d(channel*16, hc*order**2, 1, 1, 0, bias=True),
                                  )

    def forward(self, m, h):
        rh = F.interpolate(h, scale_factor=self.order)
        rm = self.pus(m)
        a1 = self.ao1(torch.cat([rm,h], 1))
        a2 = self.ao2(torch.cat([rm,a1], 1))
        a3 = self.ao3(torch.cat([rm,a2], 1))
        a4 = self.ao4(torch.cat([rm,a3], 1))
        a5 = self.ao5(torch.cat([rm,a4], 1))

        out = self.conv(a5)
        out = self.ps(out) 
        out = out + rh
        return out

class RCLoss(nn.Module):
    def __init__(self):
        super(RCLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ssim  = piq.SSIMLoss()
        
    def forward(self, x, y):
        x = torch.clip(x, 0, 1)
        loss = 0.1*self.ssim(x,y) + self.l1(x,y)
        return loss

if __name__ == '__main__':
    m = torch.randn(1,3,64,64).cuda()
    h = torch.randn(1,100,16,16).cuda()
    net = Net(3, 100, order=4).cuda()
    res = net(m,h)
    print(res.shape)
