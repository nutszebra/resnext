import sys
sys.path.append('./trainer')
import argparse
import nutszebra_cifar10
import resnext
import nutszebra_data_augmentation
import nutszebra_optimizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--load_model', '-m',
                        default=None,
                        help='trained model')
    parser.add_argument('--load_optimizer', '-o',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--load_log', '-l',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--save_path', '-p',
                        default='./',
                        help='model and optimizer will be saved every epoch')
    parser.add_argument('--epoch', '-e', type=int,
                        default=300,
                        help='maximum epoch')
    parser.add_argument('--batch', '-b', type=int,
                        default=256,
                        help='mini batch number')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu mode, put gpu id here')
    parser.add_argument('--start_epoch', '-s', type=int,
                        default=1,
                        help='start from this epoch')
    parser.add_argument('--train_batch_divide', '-trb', type=int,
                        default=4,
                        help='divid batch number by this')
    parser.add_argument('--test_batch_divide', '-teb', type=int,
                        default=4,
                        help='divid batch number by this')
    parser.add_argument('--lr', '-lr', type=float,
                        default=0.1,
                        help='leraning rate')
    parser.add_argument('--C', '-C', type=int,
                        default=2,
                        help='cardinality')
    parser.add_argument('--d', '-d', type=int,
                        default=64,
                        help='dimension')
    parser.add_argument('--multi', '-multi', type=int,
                        default=4,
                        help='multiplier of resblock')

    args = parser.parse_args().__dict__
    lr = args.pop('lr')
    C = args.pop('C')
    d = args.pop('d')
    multi = args.pop('multi')

    print('generating model')
    model = resnext.ResNext(10, C=C, d=d, multiplier=multi)
    print('Done')
    print('Parameters: {}'.format(model.count_parameters()))
    optimizer = nutszebra_optimizer.OptimizerResNext(model, lr=lr)
    args['model'] = model
    args['optimizer'] = optimizer
    args['da'] = nutszebra_data_augmentation.DataAugmentationCifar10NormalizeSmall
    main = nutszebra_cifar10.TrainCifar10(**args)
    main.run()
