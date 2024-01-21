import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
# from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, Resize, RandomRotation, RandomHorizontalFlip, RandomCrop, RandomVerticalFlip


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)#.convert('YCbCr')
    # Convert to YCbCr here
    #y, _, _ = img.split()
    #return y
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_pre = None, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.input_pre = input_pre
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        input = self.input_pre(input)
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
