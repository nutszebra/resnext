# What's this
Implementation of ResNext by chainer  

# Dependencies

    git clone https://github.com/nutszebra/resnext.git
    cd resnext
    git submodule init
    git submodule update

# How to run
    python main.py -g 0

# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for data-augmentation.  
* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  

# Cifar10 result
| network               | model detail | total accuracy (%)                          |
|:----------------------|--------------|--------------------------------------------:|
| ResNext [[1]][Paper]  | 16x64d       |96.42                                        |
| ResNext [[1]][Paper]  | 8x64d        |96.35                                        |
| ResNext [[1]][Paper]  | 2x64d        |95.98 (by my eyes on Figure 7)               |
| my implementation     | 2x64d        |95.72                                        |

<img src="https://github.com/nutszebra/resnext/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/resnext/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">


# Converted ILSVRC pretrained model
| C  | d | total layers | url | original model: ILSVRC top-1 error (%)| converted model: ILSVRC top-1 error (%)| converted model: ILSCRC top-5 error (%)|
|:---|---|--------------|-----|---------------------------------------|----------------------------------------|-----------------------------------------:|
| 64 | 4 | 101 | https://1drv.ms/u/s!AtHe5bQGa25xiIswQGB9cdcHWDUhNA  | 20.4 | 21.4 | 5.86 |
| 32 | 4 | 101 | https://1drv.ms/u/s!AtHe5bQGa25xiIsyHnhhdNNcugAqLA  | 21.2 | 22.3 | 6.24 |
| 32 | 4 | 50  | https://1drv.ms/u/s!AtHe5bQGa25xiIsxun5XuoIpd_bFjg  | 22.2 | 23.4 | 6.96 |


# How to convert ILSVRC pretrained model by yourself
To run ilsvrc_converter.py, you need to install [pytorch](http://pytorch.org/) and download t7 file from [here](https://github.com/facebookresearch/ResNeXt), then type like this on terminal:

    ipython
    run ilsvrc_converter.py -t /path/to/t7/file
    model.save_model('path/to/save/model')

Note: Please do not rename t7 file

# How to test converted ILSVRC pretrained model

    run test_ilsvrc.py -g 0 -b 16 -c 32 -d 4 -l 50 -m /path/to/converted/chainer/model -ld ./ILSVRC

g: gpu number  
b: batch number  
c: cardinality  
d: d of Cxd  
l: total layers  
m: converted model  
ld: path to root directory of ilsvrc  

# How to load converted ILSVRC pretrained model

    import resnext_ilsvrc
    import data_augmentation
    
    layers = 101 # this parameter is up to model
    C = 64 # this parameter is up to model
    
    if layers == 101 and C == 64:
        model = resnext.ResNext(1000, block_num=(3, 4, 23, 3), C=C, d=d, multiplier=1)
    elif layers == 101 and C == 32:
        model = resnext.ResNext(1000, block_num=(3, 4, 23, 3), C=C, d=d, multiplier=2)
    elif layers == 50 and C == 32:
        model = resnext.ResNext(1000, block_num=(3, 4, 6, 3), C=C, d=d, multiplier=2)
    else:
        model = resnext.ResNext(1000, C=C, d=d)
    
    model.load_model('path/to/converted/chainer/model')
    model.check_gpu(-1) # -1 means cpu. If you'd like to use gpu, give gpu id here
    
    preprocess = data_augmentation.DataAugmentationNormalizeBigger
    
    img = preprocess.test('path/to/image').x
    x = model.prepare_input([x], volatile=True)
    y = model(x, train=False)
 
# References
Aggregated Residual Transformations for Deep Neural Networks [[1]][Paper]



[paper]: https://arxiv.org/abs/1611.05431 "Paper"
