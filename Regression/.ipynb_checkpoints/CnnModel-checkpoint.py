import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable

"""
    这篇代码主要集成了四种常见的网络结构以及根据这四种结构自定义的模型
"""
ac_dict = {
    'relu':nn.ReLU,
    'lrelu':nn.LeakyReLU

}


linear1=[649,5184,5184,5184,5120,5120,5120,5120,4096,4096]
#3,4,5....
linear=[33568,33536,33536,33536,33280,32768,32768,32768]
class AlexNet(nn.Module):
    def __init__(self,acti,c_num):
        super(AlexNet, self).__init__()
        self.layers=nn.ModuleList([])
        input_channel=1
        output_channel=16
        for i in range(1,c_num):
            self.layers.append(nn.Conv1d(input_channel,output_channel,3,padding=1))
            self.layers.append(nn.BatchNorm1d(num_features=output_channel))
            self.layers.append(ac_dict[acti](inplace=True))
            self.layers.append(nn.MaxPool1d(2,2))
            input_channel=output_channel
            output_channel=output_channel*2
        #linear[c_num-1]
        self.reg = nn.Sequential(
             nn.Linear(linear[c_num-3], 15000),  #根据自己数据集修改
             nn.ReLU(inplace=True),
             nn.Linear(15000,5000),
             nn.ReLU(inplace=True),
             nn.Dropout(0.5),
             nn.Linear(5000, 1),
        )

    def forward(self,x):
        out = x
        for layer in self.layers:
            out = layer(out)         
        out = out.flatten(start_dim=1)
        # out = out.view(-1,self.output_channel)
        out = self.reg(out)
        return out

class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,out_C):
        super(Inception,self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_c, c1,kernel_size=1,padding=0),
            nn.Conv1d(c1, c1, kernel_size=3, padding=1)
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(in_c, c2,kernel_size=1,padding=0),
            nn.Conv1d(c2, c2, kernel_size=5, padding=2)

        )
        self.p3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            nn.Conv1d(in_c, c3,kernel_size=3,padding=1),
        )
        self.conv_linear = nn.Conv1d((c1+c2+c3), out_C, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_c != out_C:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_c, out_C, 1, 1, 0, bias=False),

            )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        out =  torch.cat((p1,p2,p3),dim=1)
        out += self.short_cut(x)
        return out




class DeepSpectra(nn.Module):
    def __init__(self,acti,c_num):
        super(DeepSpectra, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=3, padding=0)
        )
        self.Inception = Inception(16, 32, 32, 32, 96)
        self.fc = nn.Sequential(
            nn.Linear(20640, 5000),
            nn.Dropout(0.5),
            nn.Linear(5000, 1)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Inception(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# class InceptionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(InceptionBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
#         self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

#     def forward(self, x):
#         out1 = F.relu(self.conv1(x))
#         out2 = F.relu(self.conv2(x))
#         out3 = F.relu(self.conv3(x))
#         out4 = F.relu(self.conv4(x))
#         out = torch.cat((out1, out2, out3, out4), dim=1)
#         return out

# class DeepSpectra(nn.Module):
#     def __init__(self, num_inception_blocks=4, num_classes=1):
#         super(DeepSpectra, self).__init__()
#         self.num_inception_blocks = num_inception_blocks
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
#         self.inception_blocks = nn.ModuleList([InceptionBlock(32*(i+1), 32*(i+2)) for i in range(num_inception_blocks)])
#         self.fc1 = nn.Linear(32*(num_inception_blocks+1), 64)
#         self.fc2 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         for i in range(self.num_inception_blocks):
#             out = self.inception_blocks[i](out)
#         out = F.avg_pool1d(out, kernel_size=out.size()[2])
#         out = out.view(out.size(0), -1)
#         out = F.relu(self.fc1(out))
#         out = self.fc2(out)
#         return out
#####################
class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample,acti):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            ac_dict[acti](inplace=True),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            ac_dict[acti](inplace=True),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            ac_dict[acti](inplace=True),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual

    



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self, c_num, num_classes=1):
        super(Resnet, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(16, c_num)
        self.layer2 = self.make_layer(32, c_num, stride=2)
        self.layer3 = self.make_layer(64, c_num, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out



###############
class DenseLayer(torch.nn.Module):
    def __init__(self,in_channels,acti,middle_channels=128,out_channels=32):
        super(DenseLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            ac_dict[acti](inplace=True),
            torch.nn.Conv1d(in_channels,middle_channels,1),
            torch.nn.BatchNorm1d(middle_channels),
            ac_dict[acti](inplace=True),
            torch.nn.Conv1d(middle_channels,out_channels,3,padding=1)
        )
    def forward(self,x):
        return torch.cat([x,self.layer(x)],dim=1)


class DenseBlock(torch.nn.Sequential):

    def __init__(self,layer_num,growth_rate,in_channels,acti,middele_channels=128):
        super(DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = DenseLayer(in_channels+i*growth_rate,acti,middele_channels,growth_rate)
            self.add_module('denselayer%d'%(i),layer)

class Transition(torch.nn.Sequential):
    def __init__(self,channels,acti):
        super(Transition, self).__init__()
        self.add_module('norm',torch.nn.BatchNorm1d(channels))
        self.add_module('relu',ac_dict[acti](inplace=True))
        self.add_module('conv',torch.nn.Conv1d(channels,channels//2,3,padding=1))
        self.add_module('Avgpool',torch.nn.AvgPool1d(2))


class DenseNet(torch.nn.Module):
    def __init__(self,acti,c_num):
        super(DenseNet, self).__init__()
        layer_num=(6,12,24,16)
        growth_rate=32
        init_features=64
        middele_channels=128
        self.feature_channel_num=init_features
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(1,self.feature_channel_num,7,2,3),
            torch.nn.BatchNorm1d(self.feature_channel_num),
            ac_dict[acti](inplace=True),
            torch.nn.MaxPool1d(3,2,1),
        )
        self.DenseBlock1=DenseBlock(layer_num[0],growth_rate,self.feature_channel_num,acti,middele_channels)
        self.feature_channel_num=self.feature_channel_num+layer_num[0]*growth_rate
        self.Transition1=Transition(self.feature_channel_num,acti)
        
        self.layers=nn.ModuleList([])
        for i in range(1,c_num):
            self.layers.append(DenseBlock(layer_num[i],growth_rate,self.feature_channel_num//2,acti,middele_channels))
            self.feature_channel_num=self.feature_channel_num//2+layer_num[i]*growth_rate
            if (i!=c_num):
                self.layers.append(Transition(self.feature_channel_num,acti))
        self.layers.append(torch.nn.AdaptiveAvgPool1d(1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.feature_channel_num, self.feature_channel_num//2),
            ac_dict[acti](inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.feature_channel_num//2, 1),

        )


    def forward(self,x):
        x = self.features(x)

        x = self.DenseBlock1(x)
        x = self.Transition1(x)
        out = x
        for layer in self.layers:
            out = layer(out)    
        out = out.view(-1,self.feature_channel_num)
        out = self.fc(out)

        return out
###########################
# 对于较小的数据集和较简单的任务，可以选择较小的num_layers_per_block和growth_rate。例如，num_layers_per_block可以选择为4，growth_rate可以选择为12。
# 对于较大的数据集和较复杂的任务，可以选择更大的num_layers_per_block和growth_rate。例如，num_layers_per_block可以选择为6，growth_rate可以选择为32。
# 对于中等大小的数据集和任务，可以选择中间的值。例如，num_layers_per_block可以选择为5，growth_rate可以选择为24。
# 需要注意的是，如果您的模型太大，可能会发生过拟合，而如果您的模型太小，可能会出现欠拟合。因此，选择合适的num_layers_per_block和growth_rate是非常重要的。一般来说，您可以根据经验和交叉验证的结果来调整这些超参数，以获得最佳的性能。
