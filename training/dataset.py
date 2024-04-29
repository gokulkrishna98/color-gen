import json
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, random_split

SIZE = 512

class UnconditionalDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        source = cv2.resize(source, (SIZE, SIZE))
        target = cv2.imread(target_filename)
        target = cv2.resize(target, (SIZE, SIZE))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

dataset = UnconditionalDataset()

# Define the sizes for training, validation, and test datasets
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# print("Size of Train Dataset:", len(train_dataset))
# print("Size of Validation Dataset:", len(val_dataset))
# print("Size of Test Dataset:", len(test_dataset))