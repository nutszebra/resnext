import sys
sys.path.append('./trainer')
import nutszebra_ilsvrc_object_localization
import resnext_ilsvrc as resnext
import argparse
import trainer.nutszebra_data_augmentation as da

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--load_model', '-m',
                        default=None,
                        help='trained model')
    parser.add_argument('--load_data', '-ld',
                        default=None,
                        help='ilsvrc path')
    parser.add_argument('--save_path', '-p',
                        default='./',
                        help='model and optimizer will be saved at every epoch')
    parser.add_argument('--batch', '-b', type=int,
                        default=64,
                        help='mini batch number')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu, put gpu id here')
    parser.add_argument('--test_batch_divide', '-teb', type=int,
                        default=1,
                        help='divid test batch number by this')
    parser.add_argument('--C', '-c', type=int,
                        default=64,
                        help='C of resnext')
    parser.add_argument('--d', '-d', type=int,
                        default=4,
                        help='d of resnext')
    parser.add_argument('--layers', '-l', type=int,
                        default=101,
                        help='total layers')

    args = parser.parse_args().__dict__
    print(args)
    d = args.pop('d')
    C = args.pop('C')
    layers = args.pop('layers')

    print('generating model')
    if layers == 101 and C == 64:
        model = resnext.ResNext(1000, block_num=(3, 4, 23, 3), C=C, d=d, multiplier=1)
    elif layers == 101 and C == 32:
        model = resnext.ResNext(1000, block_num=(3, 4, 23, 3), C=C, d=d, multiplier=2)
    elif layers == 50 and C == 32:
        model = resnext.ResNext(1000, block_num=(3, 4, 6, 3), C=C, d=d, multiplier=2)
    else:
        model = resnext.ResNext(1000, C=C, d=d)
    print('Done')
    args['model'] = model
    args['da'] = da.DataAugmentationNormalizeBigger
    main = nutszebra_ilsvrc_object_localization.TrainIlsvrcObjectLocalizationClassification(**args)
    main.test_one_epoch()
