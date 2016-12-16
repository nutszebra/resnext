import six
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


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


class ResNextBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channels=(128, 128, 256), filters=(1, 3, 1), strides=(1, 1, 1), pads=(0, 1, 0), C=32):
        super(ResNextBlock, self).__init__()
        modules = []
        for i, out_channel in enumerate(out_channels[:-1]):
            for ii in six.moves.range(1, C + 1):
                modules += [('conv_bn_relu{}_{}'.format(i + 1, ii), Conv_BN_ReLU(in_channel, int(out_channels[i] / C), filters[i], strides[i], pads[i]))]
            in_channel = int(out_channels[i] / C)
        modules += [('conv_bn_relu{}'.format(len(out_channels)), Conv_BN_ReLU(out_channels[-2], out_channels[-1], filters[-1], strides[-1], pads[-1]))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.filters = filters
        self.strides = strides
        self.pads = pads
        self.C = C
        self.n = len(self.out_channels)
        self.N = six.moves.range(1, self.n)

    def info(self, indent=' ' * 4):
        for i, out_channel in enumerate(self.out_channels[:-1]):
            for ii in six.moves.range(1, self.C + 1):
                name = 'conv_bn_relu{}_{}'.format(i + 1, ii)
                print('{}{}: {}'.format(indent, name, self[name].conv.W.data.shape))
        name = 'conv_bn_relu{}'.format(len(self.out_channels))
        print('{}{}: {}'.format(indent, name, self[name].conv.W.data.shape))

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        if x_channel == h_channel:
            return x
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    def maybe_pooling(self, x):
        if 2 in self.strides:
            return F.average_pooling_2d(x, 1, 2, 0)
        return x

    def __call__(self, x, train=False):
        h = [x for _ in six.moves.range(self.C)]
        for i, out_channel in enumerate(self.out_channels[:-1]):
            h = [self['conv_bn_relu{}_{}'.format(i + 1, ii + 1)](inp, train) for ii, inp in enumerate(h)]
        h = F.concat(h, axis=1)
        h = self['conv_bn_relu{}'.format(self.n)](h, train)
        return h + ResNextBlock.concatenate_zero_pad(self.maybe_pooling(x), h.data.shape, h.volatile, type(h.data))

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count


class ResNext(nutszebra_chainer.Model):

    def __init__(self, category_num, block_num=(3, 3, 3), C=4, d=64, multiplier=4):
        super(ResNext, self).__init__()
        # conv
        modules = [('conv_bn_relu', Conv_BN_ReLU(3, C * d, 3, 1, 1))]
        out_channels = [(C * d * i, C * d * i, C * d * multiplier * i) for i in [2 ** x for x in six.moves.range(len(block_num))]]
        in_channel = C * d
        for i, n in enumerate(block_num):
            for ii in six.moves.range(n):
                if i >= 1 and ii == 0:
                    strides = (1, 2, 1)
                else:
                    strides = (1, 1, 1)
                modules += [('resnext_block_{}_{}'.format(i + 1, ii + 1), ResNextBlock(in_channel, out_channels[i], (1, 3, 1), strides, (0, 1, 0), C))]
                in_channel = out_channels[i][-1]
        modules += [('linear', Conv_BN_ReLU(in_channel, category_num, 1, 1, 0))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.category_num = category_num
        self.C = C
        self.block_num = block_num
        self.n = len(block_num)
        self.name = 'ResNext_{}_{}'.format(category_num, C)

    def info(self):
        for i, n in enumerate(self.block_num):
            for ii in six.moves.range(n):
                print('resnext_block_{}_{}'.format(i + 1, ii + 1))
                self['resnext_block_{}_{}'.format(i + 1, ii + 1)].info()
        print('{}; {}'.format('linear', self.linear.conv.W.data.shape))

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count

    def __call__(self, x, train=False):
        h = self.conv_bn_relu(x, train)
        for i, n in enumerate(self.block_num):
            for ii in six.moves.range(n):
                h = self['resnext_block_{}_{}'.format(i + 1, ii + 1)](h, train)
        batch, channels, height, width = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (height, width)), (batch, channels, 1, 1))
        return F.reshape(self.linear(h, train), (batch, self.category_num))

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
