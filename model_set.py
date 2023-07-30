import torch.nn as nn
import torch.nn.functional as F
#from esm.EGNN_Layer import GraphConvolution
#from esm.EGNN_Layer import InnerProductDecoder
import torch
from torch.nn.parameter import Parameter

class Linear_layer(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""
    def __init__(self, in_features, bias=True,act=F.sigmoid,act_true = False):
        super().__init__()
        self.regression = nn.Linear(in_features, 1, bias)
        self.normalization = torch.nn.InstanceNorm2d(in_features)
        self.act = act
        self.act_true = act_true
    def forward(self, features):
        # 将n张map乘以w+b，经过线性回归变成1张map
        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        features = self.normalization(features)
        features = features.permute(0, 2, 3, 1).squeeze(0)

        output = self.regression(features).squeeze(-1)
        score = output

        if self.act_true:
            output = self.act(output)
        return output, score # squeeze去掉维数为1的的维度

class Conv_layer3(nn.Module):
    def __init__(self, in_channel,act=F.sigmoid,act_true = False):
        super().__init__()
        self.normalization = torch.nn.InstanceNorm2d(in_channel)
        self.conv2d = nn.Conv2d(in_channel, 1, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.act = act
        self.act_true = act_true
    def forward(self, features):

        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        features = self.normalization(features)
        output = self.conv2d(features).squeeze(0).squeeze(0)

        score = output

        if self.act_true:
            output = self.act(output)

        return output, score# squeeze去掉维数为1的的维度

class Conv_layer5(nn.Module):
    def __init__(self, in_channel,act=F.sigmoid,act_true = False):
        super().__init__()
        self.normalization = torch.nn.InstanceNorm2d(in_channel)
        self.conv2d = nn.Conv2d(in_channel, 1, 5, stride=1, padding=2, dilation=1, groups=1, bias=True)
        self.act = act
        self.act_true = act_true
    def forward(self, features):

        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        features = self.normalization(features)
        output = self.conv2d(features).squeeze(0).squeeze(0)

        score = output

        if self.act_true:
            output = self.act(output)

        return output, score# squeeze去掉维数为1的的维度
class Conv_layer3_linear(nn.Module):
    def __init__(self, in_channel,hidden_size,act=F.sigmoid,act_true = False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, hidden_size, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.regression = nn.Linear(hidden_size, 1, True)
        self.act = act
        self.act_true = act_true
    def forward(self, features):
        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        output = self.conv2d(features).squeeze(0).squeeze(0)
        output1 = output.permute(1, 2, 0)
        output2 = self.regression(output1).squeeze(-1)
        if self.act_true:
            output2 = self.act(output2)
        return output2
class Conv_layer5_linear(nn.Module):
    def __init__(self, in_channel,hidden_size,act=F.sigmoid,act_true = False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, hidden_size, 5, stride=1, padding=2, dilation=1, groups=1, bias=True)
        self.regression = nn.Linear(hidden_size, 1, True)
        self.act = act
        self.act_true = act_true
    def forward(self, features):
        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        output = self.conv2d(features).squeeze(0).squeeze(0)
        output1 = output.permute(1, 2, 0)
        output2 = self.regression(output1).squeeze(-1)
        if self.act_true:
            output2 = self.act(output2)
        return output2

class BiLSTM(nn.Module):

    def __init__(self, in_channel,hidden_size,nl,act=F.sigmoid,act_true = False):
        super().__init__()
        self.bilstm = nn.LSTM(in_channel, hidden_size, num_layers=nl, bidirectional=True)
        self.regression = nn.Linear(hidden_size*2, 1, True)
        self.normalization = torch.nn.InstanceNorm2d(hidden_size*2)
        self.dropout = nn.Dropout(0.1)
        self.act = act
        self.act_true = act_true

    def forward(self, features):

        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        features = self.normalization(features)
        features = features.permute(0, 2, 3, 1).squeeze(0)

        output1,(h_n,c_n) = self.bilstm(features)

        output1 = output1.unsqueeze(0).permute(0, 3, 1, 2)
        output1 = self.normalization(output1)
        output1 = output1.permute(0, 2, 3, 1).squeeze(0)

        output = self.regression(output1).squeeze(-1)

        score = output

        if self.act_true:
            output = self.act(output)

        return output, score# squeeze去掉维数为1的的维度

class TurnBiLSTM(nn.Module):
    def __init__(self, in_channel,hidden_size,nl,act=F.sigmoid,act_true = False):
        super().__init__()
        self.bilstm = nn.LSTM(in_channel, hidden_size, num_layers=nl,bidirectional=True)
        self.normalization1 = torch.nn.InstanceNorm2d(in_channel)
        self.normalization2 = torch.nn.InstanceNorm2d(hidden_size*4)
        self.dropout = nn.Dropout(0.1)
        self.regression = nn.Linear(hidden_size*4, 1, True)
        self.act = act
        self.act_true = act_true

        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.orthogonal(m.weight)

            if isinstance(m, nn.LSTM):

                nn.init.orthogonal(m.weight_ih_l0)
                nn.init.orthogonal(m.weight_hh_l0)

    def forward(self, features):

        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        features = self.normalization1(features)
        features = features.permute(0, 2, 3, 1).squeeze(0)

        f2 = features.permute(1,0,2)
        output1,(h_n,c_n) = self.bilstm(features)
        output2, (h_n, c_n) = self.bilstm(f2)
        output3 = torch.cat((output1,output2.permute(1,0,2)),-1)

        output3 = output3.unsqueeze(0).permute(0, 3, 1, 2)
        output3 = self.normalization2(output3)
        output3 = output3.permute(0, 2, 3, 1).squeeze(0)

        output = self.regression(output3).squeeze(-1)

        score = output



        if self.act_true:
            output = self.act(output)

        return output, score # squeeze去掉维数为1的的维度



class TurnBiLSTM_serial(nn.Module):

    def __init__(self, in_channel,hidden_size,nl,act=F.sigmoid,act_true = False):
        super().__init__()
        self.bilstm = nn.LSTM(hidden_size, hidden_size, num_layers=nl,bidirectional=True)
        self.linear = nn.Linear(in_channel*4, hidden_size, bias=True)
        self.normalization1 = torch.nn.InstanceNorm2d(hidden_size)
        self.normalization2 = torch.nn.InstanceNorm2d(hidden_size*4)
        self.dropout = nn.Dropout(0.1)
        self.regression = nn.Linear(hidden_size*4, 1, True)
        self.act = act
        self.act_true = act_true
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2, x3, x4):


        x = torch.cat((x1, x2), 2)
        x = torch.cat((x, x3), 2)
        x = torch.cat((x, x4), 2)

        x = self.linear(x)
        self.relu = nn.ReLU(inplace=True)
        x = self.relu(x)

        return self.single_lstm(x)




    def single_lstm(self, features):

        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        features = self.normalization1(features)
        features = features.permute(0, 2, 3, 1).squeeze(0)

        f2 = features.permute(1,0,2)
        output1,(h_n,c_n) = self.bilstm(features)
        output2, (h_n, c_n) = self.bilstm(f2)
        output3 = torch.cat((output1,output2.permute(1,0,2)),-1)

        output3 = output3.unsqueeze(0).permute(0, 3, 1, 2)
        output3 = self.normalization2(output3)
        output3 = output3.permute(0, 2, 3, 1).squeeze(0)

        output = self.regression(output3).squeeze(-1)


        if self.act_true:
            output = self.act(output)

        return output# squeeze去掉维数为1的的维度


class TurnBiLSTM_pallel(nn.Module):

    def __init__(self, in_channel,hidden_size,nl,act=F.sigmoid,act_true = False):
        super().__init__()
        self.bilstm1 = nn.LSTM(in_channel, hidden_size, num_layers=nl,bidirectional=True)
        self.bilstm2 = nn.LSTM(40, hidden_size, num_layers=nl, bidirectional=True)
        self.normalization1 = torch.nn.InstanceNorm2d(hidden_size)
        self.normalization2 = torch.nn.InstanceNorm2d(hidden_size*16)
        self.dropout = nn.Dropout(0.1)
        self.regression = nn.Linear(hidden_size*16, 1, True)
        self.act = act
        self.act_true = act_true
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2, x3, x4):


        x1 = self.single_lstm1(x1)
        x2 = self.single_lstm1(x2)
        x3 = self.single_lstm1(x3)
        x4 = self.single_lstm2(x4)

        x = torch.cat((x1, x2), 2)
        x = torch.cat((x, x3), 2)
        x = torch.cat((x, x4), 2)

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        x = self.normalization2(x)
        x = x.permute(0, 2, 3, 1).squeeze(0)
        x = self.regression(x).squeeze(-1)

        if self.act_true:
            x = self.act(x)

        return x

    def single_lstm1(self, features):

        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        features = self.normalization1(features)
        features = features.permute(0, 2, 3, 1).squeeze(0)

        f2 = features.permute(1,0,2)
        output1,(h_n,c_n) = self.bilstm1(features)
        output2, (h_n, c_n) = self.bilstm1(f2)
        output3 = torch.cat((output1,output2.permute(1,0,2)),-1)

        return output3




    def single_lstm2(self, features):

        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        features = self.normalization1(features)
        features = features.permute(0, 2, 3, 1).squeeze(0)

        f2 = features.permute(1,0,2)
        output1,(h_n,c_n) = self.bilstm2(features)
        output2, (h_n, c_n) = self.bilstm2(f2)
        output3 = torch.cat((output1,output2.permute(1,0,2)),-1)

        return output3








        return output3# squeeze去掉维数为1的的维度


class CrossBiLSTM(nn.Module):
    def __init__(self, in_channel,hidden_size1,hidden_size2,nl,act=F.sigmoid,act_true = False):
        super().__init__()
        self.bilstm_h = nn.LSTM(in_channel, hidden_size1, num_layers=nl,bidirectional=True)
        self.bilstm_l = nn.LSTM(hidden_size1*2, hidden_size2, num_layers=nl, bidirectional=True)
        self.normalization1 = torch.nn.InstanceNorm2d(in_channel)
        self.normalization2 = torch.nn.InstanceNorm2d(hidden_size2*2)
        self.regression = nn.Linear(hidden_size2*2, 1, True)
        self.act = act
        self.act_true = act_true
        self.dropout = nn.Dropout(0.1)

        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.orthogonal(m.weight)

            if isinstance(m, nn.LSTM):

                nn.init.orthogonal(m.weight_ih_l0)
                nn.init.orthogonal(m.weight_hh_l0)

    def forward(self, features):

        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        features = self.normalization1(features)
        features = features.permute(0, 2, 3, 1).squeeze(0)

        output1,(h_n,c_n) = self.bilstm_h(features)
        output11 = output1.permute(1,0,2)

        output11 = output11.unsqueeze(0).permute(0, 3, 1, 2)
        output11 = self.normalization2(output11)
        output11 = output11.permute(0, 2, 3, 1).squeeze(0)

        output2, (h_n, c_n) = self.bilstm_l(output11)
        output2 = self.dropout(output2)


        output3 = output2.permute(1,0,2)

        output3 = output3.unsqueeze(0).permute(0, 3, 1, 2)
        output3 = self.normalization2(output3)
        output3 = output3.permute(0, 2, 3, 1).squeeze(0)

        output = self.regression(output3).squeeze(-1)

        score = output


        if self.act_true:
            output = self.act(output)

        return output, score# squeeze去掉维数为1的的维度

class CrossBiLSTM_serial(nn.Module):

    def __init__(self, hidden_size1, hidden_size2, nl, act, act_true, index_list, dim_list):

        super().__init__()

        input_dim = 0
        for i in range(len(index_list)):
            input_dim = input_dim + index_list[i] * dim_list[i]

        self.index_list = index_list

        self.bilstm_h = nn.LSTM(input_dim, hidden_size1, num_layers=nl,bidirectional=True)
        self.bilstm_l = nn.LSTM(hidden_size1*2, hidden_size2, num_layers=nl, bidirectional=True)
        self.normalization1 = torch.nn.InstanceNorm2d(input_dim)
        self.normalization2 = torch.nn.InstanceNorm2d(hidden_size2*2)
        self.regression = nn.Linear(hidden_size2*2, 1, True)
        self.act = act
        self.act_true = act_true

        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.orthogonal(m.weight)

            if isinstance(m, nn.LSTM):

                nn.init.orthogonal(m.weight_ih_l0)
                nn.init.orthogonal(m.weight_hh_l0)

    def forward(self, x1, x2, x3, x4, x5):

        x = x1

        if(self.index_list[1] > 0):
            x = torch.cat((x, x2), 2)
        if (self.index_list[2] > 0):
            x = torch.cat((x, x3), 2)
        if (self.index_list[3] > 0):
            x = torch.cat((x, x4), 2)
        if (self.index_list[4] > 0):
            x = torch.cat((x, x5), 2)

        x = self.single_lstm(x)

        output = self.regression(x).squeeze(-1)

        if self.act_true:
            output = self.act(output)

        return output  # squeeze去掉维数为1的的维度


    def single_lstm(self, features):

        features = features.unsqueeze(0).permute(0, 3, 1, 2)
        features = self.normalization1(features)
        features = features.permute(0, 2, 3, 1).squeeze(0)

        output1,(h_n,c_n) = self.bilstm_h(features)
        output11 = output1.permute(1,0,2)

        output11 = output11.unsqueeze(0).permute(0, 3, 1, 2)
        output11 = self.normalization2(output11)
        output11 = output11.permute(0, 2, 3, 1).squeeze(0)

        output2, (h_n, c_n) = self.bilstm_l(output11)
        output3 = output2.permute(1,0,2)

        output3 = output3.unsqueeze(0).permute(0, 3, 1, 2)
        output3 = self.normalization2(output3)
        output3 = output3.permute(0, 2, 3, 1).squeeze(0)

        return output3


class CrossBiLSTM_parallel(nn.Module):

    def __init__(self, hidden_size1, hidden_size2, nl, act, act_true, index_list, dim_list):

        super().__init__()

        self.index_list = index_list

        self.bilstm_h1 = nn.LSTM(dim_list[0], hidden_size1, num_layers=nl, bidirectional=True)
        self.bilstm_h2 = nn.LSTM(dim_list[1], hidden_size1, num_layers=nl, bidirectional=True)
        self.bilstm_h3 = nn.LSTM(dim_list[2], hidden_size1, num_layers=nl, bidirectional=True)
        self.bilstm_h4 = nn.LSTM(dim_list[3], hidden_size1, num_layers=nl, bidirectional=True)
        self.bilstm_h5 = nn.LSTM(dim_list[4], hidden_size1, num_layers=nl, bidirectional=True)

        self.bilstm_l = nn.LSTM(hidden_size1*2, hidden_size2, num_layers=nl, bidirectional=True)

        self.normalization1 = torch.nn.InstanceNorm2d(dim_list[0])
        self.normalization2 = torch.nn.InstanceNorm2d(dim_list[1])
        self.normalization3 = torch.nn.InstanceNorm2d(dim_list[2])
        self.normalization4 = torch.nn.InstanceNorm2d(dim_list[3])
        self.normalization5 = torch.nn.InstanceNorm2d(dim_list[4])

        self.normalization_c = torch.nn.InstanceNorm2d(hidden_size1 * 2)
        self.normalization_d = torch.nn.InstanceNorm2d(hidden_size2 * 2)

        output_dim = 0
        for i in range(len(index_list)):
            output_dim = output_dim + index_list[i] * hidden_size2 * 2

        self.regression = nn.Linear(output_dim, 1, True)
        self.act = act
        self.act_true = act_true

        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.orthogonal(m.weight)

            if isinstance(m, nn.LSTM):

                nn.init.orthogonal(m.weight_ih_l0)
                nn.init.orthogonal(m.weight_hh_l0)


    def forward(self, x1, x2, x3, x4, x5):

        x = self.single_lstm(x1, 1)

        if (self.index_list[1] > 0):
            x2 = self.single_lstm(x2, 2)
            x = torch.cat((x, x2), 2)

        if (self.index_list[2] > 0):
            x3 = self.single_lstm(x3, 3)
            x = torch.cat((x, x3), 2)

        if (self.index_list[3] > 0):
            x4 = self.single_lstm(x4, 4)
            x = torch.cat((x, x4), 2)

        if (self.index_list[4] > 0):
            x5 = self.single_lstm(x5, 5)
            x = torch.cat((x, x5), 2)

        output = self.regression(x).squeeze(-1)

        if self.act_true:
            output = self.act(output)

        return output  # squeeze去掉维数为1的的维度


    def single_lstm(self, features, index):

        features = features.unsqueeze(0).permute(0, 3, 1, 2)

        if (index == 1):
            features = self.normalization1(features)
            features = features.permute(0, 2, 3, 1).squeeze(0)
            output1, (h_n, c_n) = self.bilstm_h1(features)
        if (index == 2):
            features = self.normalization2(features)
            features = features.permute(0, 2, 3, 1).squeeze(0)
            output1, (h_n, c_n) = self.bilstm_h2(features)
        if (index == 3):
            features = self.normalization3(features)
            features = features.permute(0, 2, 3, 1).squeeze(0)
            output1, (h_n, c_n) = self.bilstm_h3(features)
        if (index == 4):
            features = self.normalization4(features)
            features = features.permute(0, 2, 3, 1).squeeze(0)
            output1, (h_n, c_n) = self.bilstm_h4(features)
        if (index == 5):
            features = self.normalization5(features)
            features = features.permute(0, 2, 3, 1).squeeze(0)
            output1, (h_n, c_n) = self.bilstm_h5(features)


        output11 = output1.permute(1,0,2)

        output11 = output11.unsqueeze(0).permute(0, 3, 1, 2)
        output11 = self.normalization_c(output11)
        output11 = output11.permute(0, 2, 3, 1).squeeze(0)

        output2, (h_n, c_n) = self.bilstm_l(output11)
        output3 = output2.permute(1,0,2)

        output3 = output3.unsqueeze(0).permute(0, 3, 1, 2)
        output3 = self.normalization_d(output3)
        output3 = output3.permute(0, 2, 3, 1).squeeze(0)

        return output3




class Resnet(nn.Module):

    def __init__(self, in_channel, out_channel, layer_number, act = F.sigmoid, act_true = False):

        super().__init__()

        self.normalization = torch.nn.InstanceNorm2d(out_channel)

        self.conv2d1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv2d2 = nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv2d3 = nn.Conv2d(out_channel, 1, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.act = act
        self.act_true = act_true

        self.layer_number = layer_number
        self.droprate = 0.1


    def forward(self, x):

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        x = self.conv2d1(x)
        x = self.normalization(x)
        x = self.relu(x)

        for i in range(self.layer_number):

            residual = x

            out = self.conv2d2(x)
            out = self.normalization(out)
            out = F.dropout(out, p=self.droprate, training=self.training)

            out = self.relu(out)

            out = self.conv2d2(out)
            out = self.normalization(out)
            out = F.dropout(out, p=self.droprate, training=self.training)

            out += residual
            out = self.relu(out)

            x = out

        x = self.conv2d3(x).squeeze(0).squeeze(0)
        score = x

        if self.act_true:
            x = self.act(x)

        return x, score

class Resnet_Serial(nn.Module):

    def __init__(self, out_channel, layer_number, act, act_true, index_list, dim_list):

        super().__init__()

        input_dim = 0
        for i in range(len(index_list)):
            input_dim = input_dim + index_list[i] * dim_list[i]

        self.index_list = index_list

        self.normalization = torch.nn.InstanceNorm2d(out_channel)

        self.conv2d1 = nn.Conv2d(input_dim, out_channel, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv2d2 = nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv2d3 = nn.Conv2d(out_channel, 1, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.act = act
        self.act_true = act_true

        self.layer_number = layer_number
        self.droprate = 0.1


    def forward(self, x1, x2, x3, x4, x5):

        x = x1

        if (self.index_list[1] > 0):
            x = torch.cat((x, x2), 2)
        if (self.index_list[2] > 0):
            x = torch.cat((x, x3), 2)
        if (self.index_list[3] > 0):
            x = torch.cat((x, x4), 2)
        if (self.index_list[4] > 0):
            x = torch.cat((x, x5), 2)

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        x = self.conv2d1(x)
        x = self.normalization(x)
        x = self.relu(x)

        for i in range(self.layer_number):

            residual = x

            out = self.conv2d2(x)
            out = self.normalization(out)
            out = F.dropout(out, p=self.droprate, training=self.training)

            out = self.relu(out)

            out = self.conv2d2(out)
            out = self.normalization(out)
            out = F.dropout(out, p=self.droprate, training=self.training)

            out += residual
            out = self.relu(out)

            x = out

        x = self.conv2d3(x).squeeze(0).squeeze(0)
        score = x

        if self.act_true:
            x = self.act(x)

        return x, score

class Resnet_Serial_linear(nn.Module):

    def __init__(self, out_channel, layer_number, act, act_true, index_list, dim_list):

        super().__init__()

        linear_dim = 64

        input_dim = 0
        for i in range(len(index_list)):
            input_dim = input_dim + index_list[i] * linear_dim

        self.index_list = index_list

        self.linear1 = nn.Linear(dim_list[0], linear_dim, bias=True)
        self.linear2 = nn.Linear(dim_list[1], linear_dim, bias=True)
        self.linear3 = nn.Linear(dim_list[2], linear_dim, bias=True)
        self.linear4 = nn.Linear(dim_list[3], linear_dim, bias=True)
        self.linear5 = nn.Linear(dim_list[4], linear_dim, bias=True)

        self.normalization = torch.nn.InstanceNorm2d(out_channel)

        self.conv2d1 = nn.Conv2d(input_dim, out_channel, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv2d2 = nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv2d3 = nn.Conv2d(out_channel, 1, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.act = act
        self.act_true = act_true

        self.layer_number = layer_number
        self.droprate = 0.1


    def forward(self, x1, x2, x3, x4, x5):

        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x3 = self.linear3(x3)
        x4 = self.linear4(x4)
        x5 = self.linear5(x5)

        x = x1

        if (self.index_list[1] > 0):
            x = torch.cat((x, x2), 2)
        if (self.index_list[2] > 0):
            x = torch.cat((x, x3), 2)
        if (self.index_list[3] > 0):
            x = torch.cat((x, x4), 2)
        if (self.index_list[4] > 0):
            x = torch.cat((x, x5), 2)

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        x = self.conv2d1(x)
        x = self.normalization(x)
        x = self.relu(x)

        for i in range(self.layer_number):

            residual = x

            out = self.conv2d2(x)
            out = self.normalization(out)
            out = F.dropout(out, p=self.droprate, training=self.training)

            out = self.relu(out)

            out = self.conv2d2(out)
            out = self.normalization(out)
            out = F.dropout(out, p=self.droprate, training=self.training)

            out += residual
            out = self.relu(out)

            x = out

        x = self.conv2d3(x).squeeze(0).squeeze(0)

        if self.act_true:
            x = self.act(x)

        return x

