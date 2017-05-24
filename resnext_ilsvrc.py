import six
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class Linear(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel):
        super(Linear, self).__init__(
            linear=L.Linear(in_channel, out_channel),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def weight_initialization(self):
        self.linear.W.data = self.weight_relu_initialization(self.linear)
        self.linear.b.data = self.bias_initialization(self.linear, constant=0)

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.linear.W.data.shape)

    def __call__(self, x, train=False):
        return self.linear(x)


class Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)

    def __call__(self, x, train=False):
        return self.conv(x)


class Group_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1), groups=32):
        super(Group_Conv, self).__init__()
        modules = []
        for i in six.moves.range(groups):
            modules += [('conv_group_{}'.format(i), Conv(in_channel, out_channel, filter_size, stride, pad))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.groups = groups

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=False):
        s = self.in_channel
        h = [self['conv_group_{}'.format(i)](x[:, i * s: i * s + s], train=train) for i in six.moves.range(self.groups)]
        return F.concat(h, axis=1)


class BN_ReLU(nutszebra_chainer.Model):

    def __init__(self, in_channel):
        super(BN_ReLU, self).__init__(
            bn=L.BatchNormalization(in_channel),
        )
        self.in_channel = in_channel

    def weight_initialization(self):
        pass

    def __call__(self, x, train=False):
        return F.relu(self.bn(x, test=not train))

    def count_parameters(self):
        return 0


class Conv_BN(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv_BN, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(out_channel),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.bn(self.conv(x), test=not train)

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class Conv_BN_ReLU(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv_BN_ReLU, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(out_channel),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return F.relu(self.bn(self.conv(x), test=not train))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class DoNothing(nutszebra_chainer.Model):

    def __init__(self):
        super(DoNothing, self).__init__()

    def __call__(self, *args, **kwargs):
        return args[0]

    def count_parameters(self):
        return 0


class ResNextBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel=64, out_channels=(256, 256, 256), filters=(1, 3, 1), strides=(1, 1, 1), pads=(0, 1, 0), C=32, skip_connection=DoNothing()):
        super(ResNextBlock, self).__init__()
        modules = []
        modules += [('conv_bn_relu_1', Conv_BN_ReLU(in_channel, out_channels[0], filters[0], strides[0], pads[0]))]
        group_width_in = int(out_channels[0] / C)
        group_width_out = int(out_channels[1] / C)
        modules += [('conv_group_2', Group_Conv(group_width_in, group_width_out, filters[1], strides[1], pads[1], C))]
        modules += [('bn_relu_3', BN_ReLU(out_channels[1]))]
        modules += [('conv_bn_4', Conv_BN(out_channels[1], out_channels[2], filters[2], strides[2], pads[2]))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.filters = filters
        self.strides = strides
        self.pads = pads
        self.C = C
        self.group_width_in = group_width_in
        self.group_width_out = group_width_out
        self.skip_connection = skip_connection

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=False):
        h = self.conv_bn_relu_1(x, train=train)
        h = self.conv_group_2(h, train=train)
        h = self.bn_relu_3(h, train=train)
        h = self.conv_bn_4(h, train=train)
        x = h + self.skip_connection(x, train=train)
        return F.relu(x)


class ResNext(nutszebra_chainer.Model):

    def __init__(self, category_num, block_num=(3, 4, 23, 3), C=64, d=4, multiplier=1):
        super(ResNext, self).__init__()
        # conv
        modules = [('conv_bn_relu', Conv_BN_ReLU(3, 64, 7, 2, 3))]
        out_channels = [(C * d * i, C * d * i, C * d * i * multiplier) for i in [2 ** x for x in six.moves.range(len(block_num))]]
        in_channel = 64
        for i, n in enumerate(block_num):
            for ii in six.moves.range(n):

                if i >= 1 and ii == 0:
                    strides = (1, 2, 1)
                    skip_connection = Conv_BN(in_channel, out_channels[i][-1], 1, 2, 0)
                    modules += [('skip_connection_{}'.format(i), skip_connection)]
                elif i == 0 and ii == 0:
                    strides = (1, 1, 1)
                    skip_connection = Conv_BN(in_channel, out_channels[i][-1], 1, 1, 0)
                    modules += [('skip_connection_{}'.format(i), skip_connection)]
                else:
                    strides = (1, 1, 1)
                    skip_connection = DoNothing()
                modules += [('resnext_block_{}_{}'.format(i, ii), ResNextBlock(in_channel, out_channels[i], (1, 3, 1), strides, (0, 1, 0), C, skip_connection=skip_connection))]
                in_channel = out_channels[i][-1]
        modules += [('linear', Linear(in_channel, category_num))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.category_num = category_num
        self.C = C
        self.block_num = block_num
        self.n = len(block_num)
        self.name = 'ResNext_{}_{}'.format(category_num, C)

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=False):
        h = self.conv_bn_relu(x, train=train)
        h = F.max_pooling_2d(h, (3, 3), (2, 2), (1, 1))
        for i, n in enumerate(self.block_num):
            for ii in six.moves.range(n):
                h = self['resnext_block_{}_{}'.format(i, ii)](h, train=train)
        batch, channels, height, width = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (height, width)), (batch, channels))
        return self.linear(h, train)

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
