import nutszebra_data_augmentation_picture
from functools import wraps
da = nutszebra_data_augmentation_picture.DataAugmentationPicture()


def reset(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        da()
        return func(self, *args, **kwargs)
    return wrapper

    @staticmethod
    @reset
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(406, 406), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationNormalizeBigger(object):

    @staticmethod
    @reset
    def train(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(224, 288)).crop_picture_randomly(1.0, sizes=(224, 224)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(256, 256), interpolation='bicubic').crop_center(sizes=(224, 224)).scale_to_one(1.0, constant=255.).fixed_normalization(1.0, each_rgb=True).convert_to_chainer_format(1.0)
        return da.x, da.info
