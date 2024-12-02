import os
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

@DATASET_REGISTRY.register()
class Texture(DatasetBase):

    dataset_dir = "texture"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        # Read the data
        train = self.read_data("train")
        test = self.read_data("val")

        # Optionally handle few-shot setup
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        super().__init__(train_x=train, val=test, test=test)

    def read_data(self, split):
        split_dir = os.path.join(self.image_dir, split)
        items = []

        if split == 'train':
            # Collect all class directories under 'train'
            class_dirs = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
            classname_to_label = {classname: idx for idx, classname in enumerate(class_dirs)}
            for class_dir in class_dirs:
                classname = class_dir  # Use the subdirectory name as the class name
                label = classname_to_label[classname]
                class_dir_path = os.path.join(split_dir, class_dir)
                for root, _, filenames in os.walk(class_dir_path):
                    for filename in filenames:
                        if not filename.startswith('.'):
                            impath = os.path.join(root, filename)
                            item = Datum(impath=impath, label=label, classname=classname)
                            items.append(item)
        elif split == 'val':
            # For the val directory, there are subdirectories 'id' and 'ood'
            class_dirs = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
            for class_dir in class_dirs:
                classname = class_dir  # 'id' or 'ood'
                if classname == 'id':
                    label = 0
                elif classname == 'ood':
                    label = 1
                else:
                    continue  # Skip any other directories
                class_dir_path = os.path.join(split_dir, class_dir)
                imnames = listdir_nohidden(class_dir_path)
                for imname in imnames:
                    impath = os.path.join(class_dir_path, imname)
                    item = Datum(impath=impath, label=label, classname=classname)
                    items.append(item)
        else:
            raise ValueError(f"Unknown split: {split}")

        return items
