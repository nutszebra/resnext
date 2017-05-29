import nutszebra_data_augmentation_picture
from functools import wraps
da = nutszebra_data_augmentation_picture.DataAugmentationPicture()


def reset(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        da()
        return func(self, *args, **kwargs)
    return wrapper


class DataAugmentationNormalizeBigger(object):

    @staticmethod
    @reset
    def train(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(224, 288)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).crop_picture_randomly(1.0, sizes=(224, 224)).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(256, 256), interpolation='bicubic').scale_to_one(1.0, constant=255.).fixed_normalization(1.0, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229), each_rgb=True).crop_center(sizes=(224, 224)).convert_to_chainer_format(1.0)
        return da.x, da.info
