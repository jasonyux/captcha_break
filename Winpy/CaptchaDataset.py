import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from captcha.image import ImageCaptcha
from tqdm import tqdm
import random, os, re
import numpy as np
from collections import OrderedDict
from PIL import Image
from pathlib import Path

IMAGE_DATA_DIR = "real/labeled"
FILE_LABEL_PATTERN = r'([0-9a-zA-Z]+)_([0-9]+)(\.png)'

class CaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, input_length, label_length, latest_first=False):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)
        self.latest_first = latest_first

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        random_str, image = self.__getimage()
        #random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length)])
        # image = to_tensor(self.generator.generate_image(random_str))
        image = to_tensor(image)
        target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)
        input_length = torch.full(size=(1, ), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)
        return image, target, input_length, target_length

    
    def __getimage(self, dir=IMAGE_DATA_DIR):
        file = None
        if self.latest_first == True:
            # this will put the EARLIEST in front
            # smallest first distribution, with last one being 1.0
            files = sorted(Path(dir).iterdir(), key=os.path.getmtime)
            distri = self.linear_distribution(len(files))
            for pox, curr_file in enumerate(files):
                if random.random() < distri[pox]:
                    file = curr_file
                    break
            if file is None:
                # not possible
                return
        else:
            files = os.listdir(dir)
            random.shuffle(files)
            file = random.choice(files)

        file = os.path.split(file)[-1]
        matchObj = re.match(FILE_LABEL_PATTERN, file)
        label = matchObj.group(1)
        image = Image.open(os.path.join(dir, file))
        if image.mode != 'RGB': # need 3 channel RGB
	        image = image.convert('RGB')
        return label, image

    def linear_distribution(self, length):
        if length == 0:
            # error
            return None
        result = [(i+1.0)/length for i in range(length)]
        return result

