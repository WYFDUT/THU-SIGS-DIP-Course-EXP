from os.path import exists, join, basename

from torchvision.transforms import Compose, ToTensor, Resize, RandomRotation, RandomHorizontalFlip, RandomCrop, RandomVerticalFlip
import PIL
from srcnn_data_utils import DatasetFromFolder

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_pre(crop_size):
    return Compose([
          RandomRotation(7, expand=False),
          RandomCrop(crop_size, pad_if_needed=True, padding_mode='reflect'),
          RandomHorizontalFlip(),
          RandomVerticalFlip(),
    ])

def input_transform(crop_size, upscale_factor):
    return Compose([
        Resize(crop_size // upscale_factor, interpolation=PIL.Image.BICUBIC),
        Resize(crop_size, interpolation=PIL.Image.BICUBIC),
        ToTensor(),
    ])


def target_transform():
    return Compose([
        ToTensor(),
    ])


def get_training_set(upscale_factor):

    root_dir = 'dataset'
    
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir, input_pre = input_pre(crop_size),
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform())


def get_test_set(upscale_factor):
    root_dir = 'dataset'
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,input_pre = input_pre(crop_size),
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform())