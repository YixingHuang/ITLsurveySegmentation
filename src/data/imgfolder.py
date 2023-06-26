import bisect
import os
import os.path

from PIL import Image
import numpy as np
import copy
from itertools import accumulate

import torch
import torch.utils.data as data
from torchvision import datasets
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, class_to_idx, file_list):
    images = []
    # print('here')
    dir = os.path.expanduser(dir)
    set_files = [line.rstrip('\n') for line in open(file_list)]
    for target in sorted(os.listdir(dir)):
        # print(target)
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    dir_file = target + '/' + fname
                    # print(dir_file)
                    if dir_file in set_files:
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
    return images


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DatasetPairedFolder(DatasetFolder):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path).convert('L')
        label = self.loader(target).convert('L')

        if self.transform is not None:
            sample = self.transform(sample)
            label = self.target_transform(label)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return sample, label # only one channel is used

    def __len__(self) -> int:
        return len(self.samples)


class ImagePairedFolder(DatasetPairedFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples


class ImageFolderTrainVal(ImagePairedFolder):
    def __init__(self, root, files_list, transform=None, target_transform=None,
                 loader=default_loader, classes=None, class_to_idx=None, imgs=None):
        """
        :param root: root path of the dataset
        :param files_list: list of filenames to include in this dataset
        :param classes: classes to include, based on subdirs of root if None
        :param class_to_idx: overwrite class to idx mapping
        :param imgs: list of image paths (under root)
        """
        if classes is None:
            assert class_to_idx is None
            classes, class_to_idx = find_classes(root)
        elif class_to_idx is None:
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        print("Creating Imgfolder with root: {}".format(root))
        # here are still image paths
        imgs = make_dataset(root, class_to_idx, files_list) if imgs is None else imgs

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: {}\nSupported image extensions are: {}".
                                format(root, ",".join(IMG_EXTENSIONS))))
        self.root = root
        self.samples = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


class ImageFolder_Subset(ImageFolderTrainVal):
    """
    Wrapper of ImageFolderTrainVal, subsetting based on indices.
    """

    def __init__(self, dataset, indices):
        self.__dict__ = copy.deepcopy(dataset).__dict__
        self.indices = indices  # Extra

    def __getitem__(self, idx):
        return super().__getitem__(self.indices[idx])  # Only return from subset

    def __len__(self):
        return len(self.indices)


class ImageFolder_Subset_ClassIncremental(ImageFolder_Subset):
    """
    ClassIncremental to only choose samples of specific label.
    Need to subclass in order to retain compatibility with saved ImageFolder_Subset objects.
    (Can't add new attributes...)
    """

    def __init__(self, imgfolder_subset, target_idx):
        """
        Subsets an ImageFolder_Subset object for only the target idx.
        :param imgfolder_subset: ImageFolder_Subset object
        :param target_idx: target int output idx
        """
        if not isinstance(imgfolder_subset, ImageFolder_Subset):
            print("Not a subset={}".format(imgfolder_subset))
            imagefolder_subset = random_split(imgfolder_subset, [len(imgfolder_subset)])[0]
            print("A subset={}".format(imagefolder_subset))

        # Creation of this object shouldn't interfere with original object
        imgfolder_subset = copy.deepcopy(imgfolder_subset)

        # Change ds classes here, to avoid any misuse
        imgfolder_subset.class_to_idx = {label: idx for label, idx in imgfolder_subset.class_to_idx.items()
                                         if idx == target_idx}
        assert len(imgfolder_subset.class_to_idx) == 1
        imgfolder_subset.classes = next(iter(imgfolder_subset.class_to_idx))

        # (path, FC_idx) => from (path, class_to_idx[class]) pairs
        orig_samples = np.asarray(imgfolder_subset.samples)
        subset_samples = orig_samples[imgfolder_subset.indices.numpy()]
        print("SUBSETTING 1 CLASS FROM DSET WITH SIZE: ", subset_samples.shape[0])

        # Filter these samples to only those with certain label
        label_idxs = np.where(subset_samples[:, 1] == str(target_idx))[0]  # indices row
        print("#SAMPLES WITH LABEL {}: {}".format(target_idx, label_idxs.shape[0]))

        # Filter the corresponding indices
        final_indices = imgfolder_subset.indices[label_idxs]

        # Sanity check
        # is first label equal to all others
        is_all_same_label = str(target_idx) == orig_samples[final_indices, 1]
        assert np.all(is_all_same_label)

        # Make a ImageFolder of the whole
        super().__init__(imgfolder_subset, final_indices)


class ImageFolder_Subset_PathRetriever(ImageFolder_Subset):
    """
    Wrapper for Imagefolder_Subset: Also returns path of the images.
    """

    def __init__(self, imagefolder_subset):
        if not isinstance(imagefolder_subset, ImageFolder_Subset):
            print("Transforming into Subset Wrapper={}".format(imagefolder_subset))
            imagefolder_subset = random_split(imagefolder_subset, [len(imagefolder_subset)])[0]
        super().__init__(imagefolder_subset, imagefolder_subset.indices)

    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolder_Subset_PathRetriever, self).__getitem__(index)
        # the image file path
        path = self.samples[self.indices[index]][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path


class ImagePathlist(data.Dataset):
    """
    Adapted from: https://github.com/pytorch/vision/issues/81
    Load images from a list with paths (no labels).
    """

    def __init__(self, imlist, targetlist=None, root='', transform=None, loader=default_loader):
        self.imlist = imlist
        self.targetlist = targetlist
        self.root = root
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]

        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        if self.targetlist is not None:
            # target = self.targetlist[index] # HYX: this is the original code
            targetpath = self.targetlist[index]

            target = self.loader(os.path.join(self.root, targetpath))
            return img, target
        else:
            return img

    def __len__(self):
        return len(self.imlist)


def random_split(dataset, lengths):
    """
    Creates ImageFolder_Subset subsets from the dataset, by altering the indices.
    :param dataset:
    :param lengths:
    :return: array of ImageFolder_Subset objects
    """
    assert sum(lengths) == len(dataset)
    indices = torch.randperm(sum(lengths))
    return [ImageFolder_Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(accumulate(lengths), lengths)]


class ConcatDatasetDynamicLabels(torch.utils.data.ConcatDataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
        the output labels are shifted by the dataset index which differs from the pytorch implementation that return the original labels
    """

    def __init__(self, datasets, classes_len, init_freeze=True):
        """
        :param datasets: List of Imagefolders
        :param classes_len: List of class lengths for each imagefolder
        """
        super(ConcatDatasetDynamicLabels, self).__init__(datasets)
        self.cumulative_classes_len = list(accumulate(classes_len))
        self.init_freeze = init_freeze

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        img, label = self.datasets[dataset_idx][idx]
        # if dataset_idx == 0:
        #     sample_idx = idx
        #     img, label = self.datasets[dataset_idx][sample_idx]
        # else:
        #     sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        #     img, label = self.datasets[dataset_idx][sample_idx]
        #     if self.init_freeze:
        #         label = label  # NO Shift Labels
        #     else:
        #         label = label + self.cumulative_classes_len[dataset_idx - 1]  # Shift Labels
        return img, label