from torchvision import transforms
import numpy as np
from torchvision.transforms.transforms import Normalize


def transform_data(data):
    transformed_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((112)),
            transforms.ToTensor(),
            transforms.Normalize(0.50, 0.50),
        ]
    )
    return transformed_train(np.array(data))
