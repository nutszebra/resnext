import six
import argparse
import resnext_ilsvrc
from torch.utils.serialization import load_lua


def judge_component(component, which):
    if which in str(type(component)):
        return True
    return False


def judge_bn(component):
    return judge_component(component, 'SpatialBatchNormalization')


def judge_linear(component):
    return judge_component(component, 'nn.Linear.Linear')


def judge_torch_obj(component):
    return judge_component(component, 'TorchObject')


def judge_relu(component):
    return judge_torch_obj(component) and 'cndnn.ReLU' in component._typename


def judge_conv(component):
    return judge_torch_obj(component) and 'SpatialConvolution' in component._typename


def extract_bn(component):
    parameters = {}
    parameters['mean'] = component.__dict__['running_mean'].numpy()
    parameters['var'] = component.__dict__['running_var'].numpy()
    parameters['gamma'] = component.__dict__['weight'].numpy()
    parameters['beta'] = component.__dict__['bias'].numpy()
    parameters['whoami'] = 'bn {}'.format(parameters['beta'].shape)
    return parameters


def extract_conv(component):
    parameters = {'dH': 1, 'padH': 0}
    if 'bias' in component._obj:
        parameters['bias'] = component._obj['bias'].numpy()
    if 'weight' in component._obj:
        parameters['weight'] = component._obj['weight'].numpy()
    if 'groups' in component._obj:
        parameters['groups'] = component._obj['groups']
    if 'dH' in component._obj:
        parameters['dH'] = component._obj['dH']
    if 'padH' in component._obj:
        parameters['padH'] = component._obj['padH']
    parameters['whoami'] = 'conv {} {} {} {}'.format(parameters['weight'].shape, parameters['groups'], parameters['dH'], parameters['padH'])
    return parameters


def extract_linear(component):
    parameters = {}
    if 'bias' in component.__dict__:
        parameters['bias'] = component.__dict__['bias'].numpy()
    if 'weight' in component.__dict__:
        parameters['weight'] = component.__dict__['weight'].numpy()
    parameters['whoami'] = 'linear {}'.format(parameters['weight'].shape)
    return parameters


def copy_linear(model, dic):
    if 'bias' in dic:
        assert model.b.data.shape == dic['bias'].shape
        assert model.b.data.dtype == dic['bias'].dtype
        model.b.data = dic['bias']
    if 'weight' in dic:
        assert model.W.data.shape == dic['weight'].shape
        assert model.W.data.dtype == dic['weight'].dtype
        model.W.data = dic['weight']


def copy_conv(model, dic):
    if 'dH' in dic:
        assert model.stride[0] == dic['dH']
    if 'padH' in dic:
        assert model.pad[0] == dic['padH']
    if 'bias' in dic:
        assert model.b.data.shape == dic['bias'].shape
        assert model.b.data.dtype == dic['bias'].dtype
        model.b.data = dic['bias']
    if 'weight' in dic:
        assert model.W.data.shape == dic['weight'].shape
        assert model.W.data.dtype == dic['weight'].dtype
        model.W.data = dic['weight']


def copy_bn(model, dic):
    if 'mean' in dic:
        assert model.avg_mean.shape == dic['mean'].shape
        assert model.avg_mean.dtype == dic['mean'].dtype
        model.avg_mean = dic['mean']
    if 'var' in dic:
        assert model.avg_var.shape == dic['var'].shape
        assert model.avg_var.dtype == dic['var'].dtype
        model.avg_var = dic['var']
    if 'gamma' in dic:
        assert model.gamma.data.shape == dic['gamma'].shape
        assert model.gamma.data.dtype == dic['gamma'].dtype
        model.gamma.data = dic['gamma']
    if 'beta' in dic:
        assert model.beta.data.shape == dic['beta'].shape
        assert model.beta.data.dtype == dic['beta'].dtype
        model.beta.data = dic['beta']


def copy_group_conv(model, dic):
    if 'groups' in dic:
        assert model.groups == dic['groups']
        if 'weight' in dic:
            branch_width = int(dic['weight'].shape[0] / dic['groups'])
            for i in six.moves.range(dic['groups']):
                m = model['conv_group_{}'.format(i)].conv
                assert m.W.data.shape == dic['weight'][i * branch_width: i * branch_width + branch_width].shape
                m.W.data = dic['weight'][i * branch_width: i * branch_width + branch_width]
        if 'bias' in dic:
            branch_width = int(dic['bias'].shape[0] / dic['groups'])
            for i in six.moves.range(dic['groups']):
                m = model['conv_group_{}'.format(i)].conv
                assert m.b.data.shape == dic['bias'][i * branch_width: i * branch_width + branch_width].shape
                m.b.data = dic['bias'][i * branch_width: i * branch_width + branch_width]


def copy_conv_bn_relu(model, parameters):
    copy_conv(model.conv, parameters[0])
    copy_bn(model.bn, parameters[1])


def copy_bn_relu(model, dic):
    copy_bn(model.bn, dic)


def copy_conv_bn(model, parameters):
    copy_conv_bn_relu(model, parameters)


def copy_resnext_block(block, parameters, skip_connection=None):
    copy_conv_bn_relu(block.conv_bn_relu_1, [parameters[0], parameters[1]])
    copy_group_conv(block.conv_group_2, parameters[2])
    copy_bn_relu(block.bn_relu_3, parameters[3])
    copy_conv_bn(block.conv_bn_4, [parameters[4], parameters[5]])
    if skip_connection is not None:
        copy_conv_bn(skip_connection, [parameters[6], parameters[7]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert torch resnext model to chainer resnext model')
    parser.add_argument('--t7', '-t',
                        default=str,
                        help='t7 file')
    args = parser.parse_args().__dict__
    t7 = args['t7']
    print(args)
    print('start convering')
    trained_model = load_lua(t7, unknown_classes=True)
    components = []

    def dfs(modules):
        try:
            for module in modules.modules:
                dfs(module)
        except:
            components.append(modules)
    dfs(trained_model)

    parameters = []
    for c in components:
        for judge, extract in ((judge_bn, extract_bn), (judge_linear, extract_linear), (judge_conv, extract_conv)):
            if judge(c):
                parameters.append(extract(c))
                break
    if '101_64x4d' in t7:
        print('101_64x4d')
        model = resnext_ilsvrc.ResNext(1000, block_num=(3, 4, 23, 3), C=64, d=4)
        count = 0
        copy_conv_bn_relu(model.conv_bn_relu, parameters[count: count + 2])
        print('conv1')
        count += 2
        for i, b in enumerate([3, 4, 23, 3]):
            print('converting block {}'.format(i))
            for ii in six.moves.range(b):
                name = 'resnext_block_{}_{}'.format(i, ii)
                print('    converting {}'.format(name))
                if ii == 0:
                    copy_resnext_block(model[name], parameters[count: count + 8], skip_connection=model['skip_connection_{}'.format(i)])
                    count += 8
                else:
                    copy_resnext_block(model[name], parameters[count: count + 6])
                    count += 6
        print('linear')
        copy_linear(model.linear.linear, parameters[count])
        count += 1
    if '101_32x4d' in t7:
        print('101_32x4d')
        model = resnext_ilsvrc.ResNext(1000, block_num=(3, 4, 23, 3), C=32, d=4, multiplier=2)
        count = 0
        copy_conv_bn_relu(model.conv_bn_relu, parameters[count: count + 2])
        print('conv1')
        count += 2
        for i, b in enumerate([3, 4, 23, 3]):
            print('converting block {}'.format(i))
            for ii in six.moves.range(b):
                name = 'resnext_block_{}_{}'.format(i, ii)
                print('    converting {}'.format(name))
                if ii == 0:
                    copy_resnext_block(model[name], parameters[count: count + 8], skip_connection=model['skip_connection_{}'.format(i)])
                    count += 8
                else:
                    copy_resnext_block(model[name], parameters[count: count + 6])
                    count += 6
        print('linear')
        copy_linear(model.linear.linear, parameters[count])
        count += 1
    if '50_32x4d' in t7:
        print('101_32x4d')
        model = resnext_ilsvrc.ResNext(1000, block_num=(3, 4, 6, 3), C=32, d=4, multiplier=2)
        count = 0
        copy_conv_bn_relu(model.conv_bn_relu, parameters[count: count + 2])
        print('conv1')
        count += 2
        for i, b in enumerate([3, 4, 6, 3]):
            print('converting block {}'.format(i))
            for ii in six.moves.range(b):
                name = 'resnext_block_{}_{}'.format(i, ii)
                print('    converting {}'.format(name))
                if ii == 0:
                    copy_resnext_block(model[name], parameters[count: count + 8], skip_connection=model['skip_connection_{}'.format(i)])
                    count += 8
                else:
                    copy_resnext_block(model[name], parameters[count: count + 6])
                    count += 6
        print('linear')
        copy_linear(model.linear.linear, parameters[count])
        count += 1
    print('finished convering')
