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

IMAGE_DATA_DIR = "real/labeled"
FILE_LABEL_PATTERN = r'([0-9a-zA-Z]+)_([0-9]+)(\.png)'

class CaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, input_length, label_length):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

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
        files = os.listdir(dir)
        file = random.choice(files)
        matchObj = re.match(FILE_LABEL_PATTERN, file)
        label = matchObj.group(1)
        image = Image.open(os.path.join(dir, file))
        if image.mode != 'RGB': # need 3 channel RGB
	        image = image.convert('RGB')
        return label, image

