import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
'''
def find_classes(dir):

    # FILE IO로 처리.
    lookup_path = os.path.join(dir, 'lookup.txt')

    classes = []
    class_to_idx = dict()
    if os.path.exists(lookup_path):
        with open( lookup_path, 'r') as rf:
            for str_line in rf:
                str_class, str_idx = str_line.split()
                classes.append(str_class)
                class_to_idx[str_class] = int(str_idx)

    return classes, class_to_idx
'''

def make_dataset(dir, train = True):

    if train == True:
        # FILE IO로 처리.
        train_path = os.path.join(dir, 'train.txt')

        image_pathes = []
        targets = []
        if os.path.exists(train_path):
            with open(train_path, 'r') as rf:
                for str_line in rf:
                    str_img_path, str_target = str_line.split()
                    str_img_path = os.path.join(dir, str_img_path)
                    image_pathes.append(str_img_path)
                    targets.append(int(str_target))

    else:
        # FILE IO로 처리.
        train_path = os.path.join(dir, 'val.txt')

        image_pathes = []
        targets = []
        if os.path.exists(train_path):
            with open(train_path, 'r') as rf:
                for str_line in rf:
                    str_img_path, str_target = str_line.split()
                    str_img_path = os.path.join(dir, str_img_path)
                    image_pathes.append(str_img_path)
                    targets.append(int(str_target))

    return image_pathes, targets


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class TINY200(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 loader=default_loader):
        # classes, class_to_idx = find_classes(root)
        imgs, targets = make_dataset(root, train)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        if self.train:
            self.train_data = []
            self.train_labels = targets

            # 배열에 PIL 이미지를 넣는다.
            for img_path in imgs:
                img = self.loader(img_path)
                self.train_data.append(img)
        else:
            self.test_data = []
            self.test_labels = targets

            # 배열에 PIL 이미지를 넣는다.
            for img_path in imgs:
                img = self.loader(img_path)
                self.test_data.append(img)


        # self.imgs = imgs
        # self.targets = targets
        # self.classes = classes
        # self.class_to_idx = class_to_idx


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # path, target = self.imgs[index]
        # img = self.loader(path)

        if self.train:
            img = self.train_data[index]
            target = self.train_labels[index]
        else:
            img = self.test_data[index]
            target = self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
