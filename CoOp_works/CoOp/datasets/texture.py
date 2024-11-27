import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

# from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class Texture(DatasetBase):

    dataset_dir = "texture"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # read images or read preprocessed data
        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            # text_file = os.path.join(self.dataset_dir, "classnames.txt")
            # classnames = self.read_classnames(text_file)

            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            # NOTE: read all data in texture as test data
            test = self.read_data_texture()

            # NOTE: make a dummy train set of 1 image with label 'OOD' (index 0)
            train = [Datum(impath=test[0].impath, label=0, classname="OOD")]

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        # load few-shot data
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        if subsample != 'all':
            # train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)    # not used
            raise NotImplementedError("Debug mode, please uncomment the last line")

        super().__init__(train_x=train, val=test, test=test)

    # @staticmethod
    # def read_classnames(text_file):
    #     """Return a dictionary containing
    #     key-value pairs of <folder name>: <class name>.
    #     """
    #     classnames = OrderedDict()
    #     with open(text_file, "r") as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             line = line.strip().split(" ")
    #             folder = line[0]
    #             classname = " ".join(line[1:])
    #             classnames[folder] = classname
    #     return classnames

    def read_data_texture(self) -> list:
        '''
        Read data for texture dataset (OOD data)
        Use label 'OOD' (index 0) for all images
        return a list of Datum objects (image path, label, classname)
        '''
        if not os.path.exists(self.image_dir) or not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Directory {self.image_dir} does not exist or is not a directory")

        folders = sorted(f for f in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, f)))  # class names (folder names)
        items = []

        for folder in folders:    # iterate through class folders
            imnames = listdir_nohidden(os.path.join(self.image_dir, folder))    # get all image names in that class folder
            classname = 'OOD'
            label = 0
            for imname in imnames:  # for each image in that class folder
                impath = os.path.join(self.image_dir, folder, imname)   # get the image path, e.g. 'texture/images/Brick/Brick_001.jpg'
                item = Datum(impath=impath, label=label, classname=classname)   # get a Datum object with image path, label, and classname
                items.append(item)  # add to list

        return items

# test with dummy cfg
if __name__ == '__main__':
    cfg = type('cfg', (object,), {})()
    cfg.DATASET = type('DATASET', (object,), {})()
    cfg.DATASET.ROOT = 'DATA'
    cfg.DATASET.NUM_SHOTS = 10
    cfg.DATASET.SUBSAMPLE_CLASSES = 'all'
    cfg.SEED = 0
    dataset = Texture(cfg)
    print(f"Number of training samples: {len(dataset.train_x)}")
    print(f"Number of test samples: {len(dataset.test)}")
    print(f"Number of classes: {dataset.num_classes}")

