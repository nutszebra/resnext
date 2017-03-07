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

# References
Aggregated Residual Transformations for Deep Neural Networks [[1]][Paper]

[paper]: https://arxiv.org/abs/1611.05431 "Paper"
