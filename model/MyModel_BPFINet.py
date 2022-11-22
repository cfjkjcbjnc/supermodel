import torch
from torch import nn
import torch.nn.functional as F
from attention import ChannelAttention as CA
from deeplab_resnet import resnet50

config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [128, 256, 256, 512, 512]],
                 'score': 128}


class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False),
                                    nn.BatchNorm2d(list_k[1][i]),
                                    nn.ReLU(inplace=True)))
        self.convert0 = nn.ModuleList(up)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl


class BasicConv(nn.Module):
    def __init__(self, channel, stride, padding=1, dilate=1):
        super(BasicConv, self).__init__()
        self.channel = channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, 3, stride=stride, padding=padding, dilation=dilate, bias=False),
            nn.BatchNorm2d(self.channel),)
            # nn.ReLU()

    def forward(self, x):
        return self.conv(x)


class USRM3(nn.Module):
    def __init__(self, channel):
        super(USRM3, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)

        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev1(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev2(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum(F.relu(x + y))


class USRM4(nn.Module):
    def __init__(self, channel):
        super(USRM4, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 2, 1, 1)
        self.conv4 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev3 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x, gi):
        # gi means global information
        y1 = self.conv1(x)
        y1 = y1 + F.interpolate(gi, y1.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)

        y3up = F.interpolate(y4, y3.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv_rev1(y3 + y3up)
        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev2(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev3(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum(F.relu(x + y))


class USRM5(nn.Module):
    def __init__(self, channel):
        super(USRM5, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 2, 1, 1)
        self.conv4 = BasicConv(self.channel, 2, 1, 1)
        self.conv5 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev3 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev4 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x, high, gi):
        # gi means global information
        y1 = self.conv1(x)
        y1 = y1 + F.interpolate(high, y1.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv2(y1)
        y2 = y2 + F.interpolate(gi, y2.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)

        y4up = F.interpolate(y5, y4.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv_rev1(y4 + y4up)
        y3up = F.interpolate(y4, y3.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv_rev2(y3 + y3up)
        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev3(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev4(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum(F.relu(x + y))


class USRM5_2(nn.Module):
    def __init__(self, channel):
        super(USRM5_2, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv(self.channel, 2, 1, 1)
        self.conv2 = BasicConv(self.channel, 2, 1, 1)
        self.conv3 = BasicConv(self.channel, 2, 1, 1)
        self.conv4 = BasicConv(self.channel, 2, 1, 1)
        self.conv5 = BasicConv(self.channel, 1, 2, 2)

        self.conv_rev1 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev2 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev3 = BasicConv(self.channel, 1, 1, 1)
        self.conv_rev4 = BasicConv(self.channel, 1, 1, 1)

        self.conv_sum = BasicConv(self.channel, 1, 1, 1)

    def forward(self, x, high, gi):
        # gi means global information
        y1 = self.conv1(x)
        y1 = y1 + F.interpolate(high, y1.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y3 = y3 + F.interpolate(gi, y3.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)

        y4up = F.interpolate(y5, y4.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv_rev1(y4 + y4up)
        y3up = F.interpolate(y4, y3.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv_rev2(y3 + y3up)
        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev3(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev4(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum(F.relu(x + y))


class ScoreLayers(nn.Module):
    def __init__(self, channel_list):
        super(ScoreLayers, self).__init__()
        self.channel_list = channel_list
        scores = []
        for channel in self.channel_list:
            scores.append(nn.Conv2d(channel, 1, 1, 1))
        self.scores = nn.ModuleList(scores)

    def forward(self, x, x_size=None):
        for i in range(len(x)):
            x[i] = self.scores[i](x[i])
        if x_size is not None:
            for i in range(len(x)):
                x[i] = F.interpolate(x[i], x_size[2:], mode='bilinear', align_corners=True)
        return x


def extra_layer(base_model_cfg, resnet):
    config = config_resnet
    convert_layers, score_layers = [], []
    convert_layers = ConvertLayer(config['convert'])
    score_layers = ScoreLayers(config['convert'][1])
    return resnet, convert_layers, score_layers


class BPFINet(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, score_layers):
        super(BPFINet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base
        self.score = score_layers
        self.config = config_resnet
        self.convert = convert_layers
        self.usrm3_1 = USRM3(self.config['convert'][1][4])
        self.usrm3_2 = USRM3(self.config['convert'][1][3])
        self.usrm4 = USRM4(self.config['convert'][1][2])
        self.usrm5_1 = USRM5(self.config['convert'][1][1])
        self.usrm5_2 = USRM5_2(self.config['convert'][1][0])

        self.ca43 = CA(self.config['convert'][1][3], self.config['convert'][1][2])
        self.ca42 = CA(self.config['convert'][1][3], self.config['convert'][1][1])
        self.ca41 = CA(self.config['convert'][1][3], self.config['convert'][1][0])
        self.ca32 = CA(self.config['convert'][1][2], self.config['convert'][1][1])
        self.ca21 = CA(self.config['convert'][1][1], self.config['convert'][1][0])

    def forward(self, x):
        x_size = x.size()
        C1, C2, C3, C4, C5 = self.base(x)
        if self.base_model_cfg == 'resnet':
            C1, C2, C3, C4, C5 = self.convert([C1, C2, C3, C4, C5])

        C5 = self.usrm3_1(C5)
        C5 = F.interpolate(C5, C4.shape[2:], mode='bilinear', align_corners=True)
        C4 = self.usrm3_2(C4 + C5)
        C4_att_3 = self.ca43(C4)
        C4_att_2 = self.ca42(C4)
        C4_att_1 = self.ca41(C4)
        C3 = self.usrm4(C3, C4_att_3)
        C3_att_2 = self.ca32(C3)
        C2 = self.usrm5_1(C2, C3_att_2, C4_att_2)
        C2_att_1 = self.ca21(C2)
        C1 = self.usrm5_2(C1, C2_att_1, C4_att_1)

        C1, C2, C3, C4, C5 = self.score([C1, C2, C3, C4, C5], x_size)
        return C1, C2, C3, C4, C5


class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AFR(nn.Module):
    def __init__(self, block, planes, blocks, stride=1):
        super(AFR, self).__init__()

        self.inplanes=32
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        self.AFR=nn.Sequential(*layers)
    def forward(self,x):
        return self.AFR(x)


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()

        self.feature=BPFINet()
        # self.feature=U_structure()
        self.AFR=AFR(block=SEBasicBlock, planes=30, blocks=1)


    def forward(self,x):

        x=torch.cat([self.feature1(x[:,0:3]),self.feature2(x[:,3:6])],dim=1)

        # x=self.feature(x)
        x=torch.squeeze(x,dim=2)
        x=self.AFR(x)

        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d_model=96
        self.nhead=4
        self.num_layers=1
        self.afr_reduced_cnn_size=30

        self.feature=feature_extraction()
        self.linear0=nn.Sequential(nn.Linear(3000, 96), nn.ReLU(True))
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.nhead, batch_first=True, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # self.autoformer=Autoformer()
        self.linear1 = nn.Sequential(nn.Linear(2880, 32), nn.ReLU(True))
        self.linear2 = nn.Sequential(nn.Linear(32, 5), nn.Softmax(dim=1))

    def forward(self,x):


        x=torch.cat((x,torch.fft.fft(x).real),dim=1)

        #without autoformer
        x_afr=self.feature(x)
        # encoded_features = self.transformer_encoder(x_afr.view(x_afr.shape[0], -1, 96))
        x_afr=self.linear0(x_afr)
        encoded_features = self.transformer_encoder(x_afr)
        # print(encoded_features.shape)

        encoded_features = x_afr * encoded_features
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        # print(encoded_features.shape)
        x_final = self.linear1(encoded_features)
        x_final = self.linear2(x_final)
        return x_final

        # # with autoformer
        # x_afr=self.feature(x)
        # # encoded_features = self.transformer_encoder(x_afr.view(x_afr.shape[0], -1, 96))
        # x_afr=self.linear0(x_afr)
        # x_afr=self.autoformer(x_afr)
        # # print(encoded_features.shape)
        # x_final = self.linear1(x_afr)
        # x_final = self.linear2(x_final)
        # return x_final

# net=MyModel()
# net.eval()
# x=torch.rand((1,6,3000))
# y=net(x)
# print(y)