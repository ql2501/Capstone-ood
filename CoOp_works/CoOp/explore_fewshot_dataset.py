import os
import pickle
import argparse

parser = argparse.ArgumentParser()
# examples like: /DATA/caltech-101/split_fewshot/shot_8-seed_1.pkl
parser.add_argument("fewshot_dataset_path", type=str, help="the few shot dataset to parse")
args = parser.parse_args()

fewshot_dataset_path = args.fewshot_dataset_path
assert os.path.exists(fewshot_dataset_path)
print("Opening file at ", fewshot_dataset_path)

# Open the file in binary read mode
with open(fewshot_dataset_path, 'rb') as file:
    data = pickle.load(file)

# Now, `data` contains the deserialized Python object
print("The fewshot dataset has: ", data.keys())
print("The training data contains ", len(data['train']), " samples")
print("The val data contains ", len(data['val']), " samples")
print()
print("Printing the 1st data's attributes: ")
print(data['train'][0].impath)
print(data['train'][0].label)
print(data['train'][0].domain)
print(data['train'][0].classname)
print()
print("Printing the 8th data's attributes: ")
print(data['train'][8].impath)
print(data['train'][8].label)
print(data['train'][8].domain)
print(data['train'][8].classname)