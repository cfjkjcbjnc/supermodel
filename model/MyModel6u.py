## Attention based MultiModal Sleep Staging Network

from re import T
from turtle import forward
import torch
from torch._C import TensorType
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
from torch.nn import init
from torchvision import transforms

from model.BasicModel import BasicModel
from functools import reduce

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class creat_bn_conv(nn.Module):
    def __init__(self, input_size, filter_size, kernel_size, padding, dilation):
        super(creat_bn_conv, self).__init__()
        self.conv = nn.Conv2d(input_size, filter_size, kernel_size=(1, kernel_size), stride=(1, 1), padding=padding,
                              dilation=(dilation, dilation))
        self.bn = nn.BatchNorm2d(filter_size)

    def forward(self, x):
        x_conv = self.conv(x)
        # print("x_conv shape:",x_conv.shape)
        x_bn = self.bn(x_conv)
        return x_bn


class creat_u_encoder(nn.Module):
    def __init__(self,
                 input,
                 final_filter,
                 kernel_size,
                 pooling_size,
                 middle_layer_filter,
                 depth,
                 padding
                 ):
        super(creat_u_encoder, self).__init__()
        self.depth = depth
        self.creat_bn_conv0 = creat_bn_conv(input_size=input, filter_size=final_filter, kernel_size=kernel_size,
                                            padding='same', dilation=1)  # final_layer
        self.creat_bn_conv1 = nn.ModuleList()
        self.creat_bn_conv1.append(
            creat_bn_conv(input_size=final_filter, filter_size=middle_layer_filter, kernel_size=kernel_size,
                          padding='same', dilation=1))
        self.creat_bn_conv1.append(
            creat_bn_conv(input_size=middle_layer_filter, filter_size=middle_layer_filter, kernel_size=kernel_size,
                          padding='same', dilation=1))
        self.creat_bn_conv1.append(
            creat_bn_conv(input_size=middle_layer_filter, filter_size=middle_layer_filter, kernel_size=kernel_size,
                          padding='same', dilation=1))

        self.creat_bn_conv2 = creat_bn_conv(input_size=middle_layer_filter, filter_size=middle_layer_filter,
                                            kernel_size=kernel_size, padding='same', dilation=1)
        self.creat_bn_conv3 = nn.ModuleList()
        self.creat_bn_conv3.append(
            creat_bn_conv(input_size=middle_layer_filter * 2, filter_size=final_filter, kernel_size=kernel_size,
                          padding='same', dilation=1))
        self.creat_bn_conv3.append(
            creat_bn_conv(input_size=middle_layer_filter * 2, filter_size=middle_layer_filter, kernel_size=kernel_size,
                          padding='same', dilation=1))
        self.creat_bn_conv3.append(
            creat_bn_conv(input_size=middle_layer_filter * 2, filter_size=middle_layer_filter, kernel_size=kernel_size,
                          padding='same', dilation=1))

        self.maxpool = nn.MaxPool2d(kernel_size=(1, pooling_size), padding=0)  # padding=(pooling_size//2,0))
        # self.upsample = transforms.Resize()#mode:'bilinear''nearest'

    def forward(self, input):
        from_encoder = []
        conv_bn0 = self.creat_bn_conv0(input)
        # print('conv_bn0.shape',conv_bn0.shape)
        conv_bn = conv_bn0
        for d in range(self.depth - 1):
            # print(d)
            conv_bn = self.creat_bn_conv1[d](conv_bn)
            # print(f'conv_bn{d}.shape',conv_bn.shape)
            from_encoder.append(conv_bn)
            # print('conv_bn',conv_bn.shape)
            if d != self.depth - 2:
                conv_bn = self.maxpool(conv_bn)
        # print("conv_bn:",conv_bn.shape)
        conv_bn = self.creat_bn_conv2(conv_bn)

        for d in range(self.depth - 1, 0, -1):
            # print('conv_bn.shape up',conv_bn.shape)
            conv_bn = transforms.Resize([from_encoder[-1].shape[2], from_encoder[-1].shape[3]])(conv_bn)
            # print('conv_bn.shape up',conv_bn.shape)
            x_concat = torch.cat((conv_bn, from_encoder.pop()), dim=1)
            # print(d)
            conv_bn = self.creat_bn_conv3[d - 1](x_concat)
        # print('conv_bn,conv_bn0',conv_bn.shape,conv_bn0.shape)
        x_final = torch.add(conv_bn, conv_bn0)
        return x_final


class create_mse(nn.Module):
    def __init__(self,
                 input,
                 final_filter,
                 kernel_size,
                 dilation_rates):
        super(create_mse, self).__init__()
        self.dilation_rates = dilation_rates
        self.creat_bn_conv0 = nn.ModuleList()
        self.creat_bn_conv0.append(
            creat_bn_conv(input_size=input, filter_size=final_filter, kernel_size=kernel_size, padding='same',
                          dilation=self.dilation_rates[0]))
        self.creat_bn_conv0.append(
            creat_bn_conv(input_size=input, filter_size=final_filter, kernel_size=kernel_size, padding='same',
                          dilation=self.dilation_rates[1]))
        self.creat_bn_conv0.append(
            creat_bn_conv(input_size=input, filter_size=final_filter, kernel_size=kernel_size, padding='same',
                          dilation=self.dilation_rates[2]))
        self.creat_bn_conv0.append(
            creat_bn_conv(input_size=input, filter_size=final_filter, kernel_size=kernel_size, padding='same',
                          dilation=self.dilation_rates[3]))  # final_layer

        self.feature = nn.Sequential(
            nn.Conv2d(final_filter * len(self.dilation_rates), final_filter * 2, kernel_size=(1, kernel_size),
                      stride=(1, 1), padding='same'),
            nn.Conv2d(final_filter * 2, final_filter, kernel_size=(1, kernel_size), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(final_filter)
        )

    def forward(self, x):
        # print("mse_x:",x.shape)
        convs = []
        for (i, dr) in enumerate(self.dilation_rates):
            conv_bn = self.creat_bn_conv0[i](x)
            convs.append(conv_bn)
        # print('con_conv_0.shape',convs.shape)
        con_conv = reduce(lambda l, r: torch.cat([l, r], dim=1), convs)
        # print('con_conv.shape',con_conv.shape)
        out = self.feature(con_conv)
        return out


class U_structure(BasicModel):
    def __init__(self):
        super(U_structure, self).__init__()
        self.padding = 'same'
        self.sleep_epoch_length = 3000
        self.sequence_length = 20
        self.filters = [16, 32, 64, 128, 256]
        self.kernel_size = 5
        self.pooling_sizes = [10, 6, 4, 2]
        self.dilation = [1, 2, 3, 4]

        self.u_depths = [4, 4, 4, 4]
        self.u_inner_filter = 8
        self.mse_filters = [32, 24, 46, 8, 5]
        self.relu = nn.ReLU(inplace=True)

        # encoder1
        self.creat_bn_conv_u1_EEG = creat_u_encoder(input=1,
                                                    final_filter=self.filters[0],
                                                    kernel_size=self.kernel_size,
                                                    pooling_size=self.pooling_sizes[0],
                                                    middle_layer_filter=self.u_inner_filter,
                                                    depth=self.u_depths[0],
                                                    padding=self.padding)
        # self.creat_bn_conv_u1_EOG= creat_u_encoder(input_size=1,filter_size=self.filters[0],kernel_size=self.kernel_size,padding=self.padding,dilation=self.dilation[0])
        self.u1 = nn.Sequential(

            nn.Conv2d(self.filters[0], int(self.filters[0] / 2), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU()

        )

        # encoder2
        self.creat_bn_conv_u2 = creat_u_encoder(
            input=int(self.filters[0] / 2),
            final_filter=self.filters[1],
            kernel_size=self.kernel_size,
            pooling_size=self.pooling_sizes[1],
            middle_layer_filter=self.u_inner_filter,
            depth=self.u_depths[1],
            padding=self.padding)

        self.u2 = nn.Sequential(

            nn.Conv2d(self.filters[1], int(self.filters[1] / 2), kernel_size=(1, 1), stride=(1, 1),
                      padding=self.padding),
            nn.ReLU(),

        )
        # encoder3
        self.creat_bn_conv_u3 = creat_u_encoder(input=int(self.filters[1] / 2),
                                                final_filter=self.filters[2],
                                                kernel_size=self.kernel_size,
                                                pooling_size=self.pooling_sizes[2],
                                                middle_layer_filter=self.u_inner_filter,
                                                depth=self.u_depths[2],
                                                padding=self.padding)

        self.u3 = nn.Sequential(

            nn.Conv2d(self.filters[2], int(self.filters[2] / 2), kernel_size=(1, 1), stride=(1, 1),
                      padding=self.padding),
            nn.ReLU(),

        )
        # encoder4
        self.creat_bn_conv_u4 = creat_u_encoder(input=int(self.filters[2] / 2),
                                                final_filter=self.filters[3],
                                                kernel_size=self.kernel_size,
                                                pooling_size=self.pooling_sizes[3],
                                                middle_layer_filter=self.u_inner_filter,
                                                depth=self.u_depths[3],
                                                padding=self.padding)

        self.u4 = nn.Sequential(

            nn.Conv2d(self.filters[3], int(self.filters[3] / 2), kernel_size=(1, 1), stride=(1, 1),
                      padding=self.padding),
            nn.ReLU(),

        )
        # encoder5
        self.creat_bn_conv_u5 = creat_u_encoder(input=int(self.filters[3] / 2),
                                                final_filter=self.filters[4],
                                                kernel_size=self.kernel_size,
                                                pooling_size=self.pooling_sizes[3],
                                                middle_layer_filter=self.u_inner_filter,
                                                depth=self.u_depths[3],
                                                padding=self.padding)

        self.u5 = nn.Sequential(
            nn.Conv2d(self.filters[4], int(self.filters[4] / 2), kernel_size=(1, 1), stride=(1, 1),
                      padding=self.padding),
            nn.ReLU()
        )

        # MES
        self.create_mse1 = create_mse(int(self.filters[0] / 2), self.mse_filters[0], self.kernel_size, self.dilation)
        self.create_mse2 = create_mse(int(self.filters[1] / 2), self.mse_filters[1], self.kernel_size, self.dilation)
        self.create_mse3 = create_mse(int(self.filters[2] / 2), self.mse_filters[2], self.kernel_size, self.dilation)
        self.create_mse4 = create_mse(int(self.filters[3] / 2), self.mse_filters[3], self.kernel_size, self.dilation)
        self.create_mse5 = create_mse(int(self.filters[4] / 2), self.mse_filters[4], self.kernel_size, self.dilation)
        # decoder4

        self.creat_u_encoder_d4 = creat_u_encoder(13, self.filters[3], self.kernel_size, self.pooling_sizes[3],
                                                  self.u_inner_filter, depth=self.u_depths[3], padding=self.padding)
        self.d4 = nn.Conv2d(self.filters[3], self.filters[3] // 2, kernel_size=(1, 1), stride=(1, 1),
                            padding=self.padding)
        # decoder3
        self.creat_u_encoder_d3 = creat_u_encoder(110, self.filters[2], self.kernel_size, self.pooling_sizes[2],
                                                  self.u_inner_filter, depth=self.u_depths[2], padding=self.padding)
        self.d3 = nn.Conv2d(self.filters[2], self.filters[2] // 2, kernel_size=(1, 1), stride=(1, 1),
                            padding=self.padding)

        # decoder2
        self.creat_u_encoder_d2 = creat_u_encoder(56, self.filters[1], self.kernel_size, self.pooling_sizes[1],
                                                  self.u_inner_filter, depth=self.u_depths[1], padding=self.padding)
        self.d2 = nn.Conv2d(self.filters[1], self.filters[1] // 2, kernel_size=(1, 1), stride=(1, 1),
                            padding=self.padding)
        # decoder1
        self.creat_u_encoder_d1 = creat_u_encoder(48, self.filters[0], self.kernel_size, self.pooling_sizes[0],
                                                  self.u_inner_filter, depth=self.u_depths[0], padding=self.padding)
        self.d1 = nn.Conv2d(self.filters[0], self.filters[0] // 2, kernel_size=(1, 1), stride=(1, 1),
                            padding=self.padding)

        # self.zero = nn.ZeroPad2d(int((self.sleep_epoch_length - int(self.filters[0]/2)) // 2))

    def forward(self, x):
        # EEG
        x_EEG = x.unsqueeze(dim=2)
        # x_EEG = x_EEG.unsqueeze(dim=2)
        # print(x_EEG.shape)
        # encoder
        x_EEG_u1 = self.creat_bn_conv_u1_EEG(x_EEG)
        # print('x_EEG_u1.shape',x_EEG_u1.shape)
        x_EEG_u1 = self.u1(x_EEG_u1)
        x_EEG_u1_pool = nn.MaxPool2d(kernel_size=(1, self.pooling_sizes[0]))(x_EEG_u1)

        # print('x_EEG_u1.shape2',x_EEG_u1.shape)
        x_EEG_u2 = self.creat_bn_conv_u2(x_EEG_u1_pool)
        # print('x_EEG_u2.shape2',x_EEG_u2.shape)
        x_EEG_u2 = self.u2(x_EEG_u2)
        x_EEG_u2_pool = nn.MaxPool2d(kernel_size=(1, self.pooling_sizes[1]))(x_EEG_u2)

        x_EEG_u3 = self.creat_bn_conv_u3(x_EEG_u2_pool)
        # print('x_EEG_u3.shape2',x_EEG_u3.shape)
        x_EEG_u3 = self.u3(x_EEG_u3)
        x_EEG_u3_pool = nn.MaxPool2d(kernel_size=(1, self.pooling_sizes[2]))(x_EEG_u3)
        #
        # print(x_EEG_u3.shape)

        x_EEG_u4 = self.creat_bn_conv_u4(x_EEG_u3_pool)
        # print('x_EEG_u4.shape2',x_EEG_u4.shape)
        x_EEG_u4 = self.u4(x_EEG_u4)
        x_EEG_u4_pool = nn.MaxPool2d(kernel_size=(1, self.pooling_sizes[3]))(x_EEG_u4)
        #
        x_EEG_u5 = self.creat_bn_conv_u5(x_EEG_u4_pool)
        # print('x_EEG_u5.shape2',x_EEG_u5.shape)
        x_EEG_u5 = self.u5(x_EEG_u5)
        #
        # MSE
        # print(x_EEG_u1.shape, x_EEG_u2.shape,x_EEG_u3.shape,x_EEG_u4.shape,x_EEG_u5.shape)
        x_EEG_u1 = self.create_mse1(x_EEG_u1)
        x_EEG_u2 = self.create_mse2(x_EEG_u2)
        x_EEG_u3 = self.create_mse3(x_EEG_u3)
        x_EEG_u4 = self.create_mse4(x_EEG_u4)
        x_EEG_u5 = self.create_mse5(x_EEG_u5)
        # print('x_EEG_u1.shape, x_EEG_u2.shape,_EEG_u3.shape,x_EEG_u4.shape,x_EEG_u5.shape',x_EEG_u1.shape, x_EEG_u2.shape,x_EEG_u3.shape,x_EEG_u4.shape,x_EEG_u5.shape)
        # #decoder
        # x_EEG_up4 = nn.functional.interpolate(x_EEG_u5,(x_EEG_u4.shape[2],x_EEG_u4.shape[3]))
        x_EEG_up4 = transforms.Resize([x_EEG_u4.shape[2], x_EEG_u4.shape[3]])(x_EEG_u5)
        # print('x_EEG_up4,x_EEG_u4',x_EEG_up4.shape,x_EEG_u4.shape)
        x_EEG_d4 = torch.cat((x_EEG_up4, x_EEG_u4), dim=1)
        # print('x_EEG_d4',x_EEG_d4.shape)
        x_EEG_d4 = self.creat_u_encoder_d4(x_EEG_d4)
        # print('x_EEG_d41',x_EEG_d4.shape)
        x_EEG_d4 = self.d4(x_EEG_d4)
        # print('x_EEG_d42',x_EEG_d4.shape)

        x_EEG_up3 = transforms.Resize([x_EEG_u3.shape[2], x_EEG_u3.shape[3]])(x_EEG_d4)
        # print('x_EEG_up3,x_EEG_u3',x_EEG_up3.shape,x_EEG_u3.shape)
        x_EEG_d3 = torch.cat((x_EEG_up3, x_EEG_u3), dim=1)
        # print('x_EEG_d3',x_EEG_d3.shape)
        x_EEG_d3 = self.creat_u_encoder_d3(x_EEG_d3)
        x_EEG_d3 = self.d3(x_EEG_d3)

        x_EEG_up2 = transforms.Resize([x_EEG_u2.shape[2], x_EEG_u2.shape[3]])(x_EEG_d3)
        x_EEG_d2 = torch.cat((x_EEG_up2, x_EEG_u2), dim=1)
        # print('x_EEG_d2',x_EEG_d2.shape)
        x_EEG_d2 = self.creat_u_encoder_d2(x_EEG_d2)
        x_EEG_d2 = self.d2(x_EEG_d2)

        x_EEG_up1 = transforms.Resize([x_EEG_u1.shape[2], x_EEG_u1.shape[3]])(x_EEG_d2)
        x_EEG_d1 = torch.cat((x_EEG_up1, x_EEG_u1), dim=1)
        # print('x_EEG_d1',x_EEG_d1.shape)
        x_EEG_d1 = self.creat_u_encoder_d1(x_EEG_d1)
        # x_EEG_d1 = self.d1(x_EEG_d1)
        # print('x_EEG_d1.shape',x_EEG_d1.shape)
        # pad2d = ((self.sleep_epoch_length * self.sequence_length - x_EEG_d1.shape[3]) // 2,
        #          (self.sleep_epoch_length * self.sequence_length - x_EEG_d1.shape[3]) // 2, 0, 0)
        # x_EEG_d1 = F.pad(x_EEG_d1, pad2d)

        # x_EEG_d1.shape=(1,16,1,3000)
        return x_EEG_d1


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

        self.inplanes=96
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
        self.EEG_feature1=U_structure()
        self.EEG_feature2=U_structure()
        self.EOG_feature1=U_structure()
        self.EOG_feature2=U_structure()
        self.EMG_feature1=U_structure()
        self.EMG_feature2=U_structure()

        # self.feature=U_structure()
        self.AFR=AFR(block=SEBasicBlock, planes=30, blocks=1)


    def forward(self,x):

        x=torch.cat([self.EEG_feature1(x[:,0:1]),
                     self.EOG_feature1(x[:,1:2]),
                     self.EMG_feature1(x[:,2:3]),
                     self.EEG_feature2(x[:,3:4]),
                     self.EOG_feature2(x[:,4:5]),
                     self.EMG_feature2(x[:,5:6])],dim=1)

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
        # feature
        self.feature=feature_extraction()
        # attention
        self.linear0=nn.Sequential(nn.Linear(3000, 96), nn.ReLU(True))
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, nhead=self.nhead, batch_first=True, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # classification
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
