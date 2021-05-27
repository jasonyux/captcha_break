import random
import torch, os, string
from PIL import Image
from torchvision.transforms.functional import to_tensor

LABEL_LENGTH = 4
CHARACTERS = string.digits + string.ascii_uppercase + string.ascii_lowercase

def decode(sequence):
    a = ''.join([CHARACTERS[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != CHARACTERS[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != CHARACTERS[0] and s[-1] != a[-1]:
        s += a[-1]
    return s

# to have the same return format as __getitem__
def prepros_image(image_path):
    image = Image.open(image_path)
    image = to_tensor(image)
    target_length = torch.full(size=(1, ), fill_value=LABEL_LENGTH, dtype=torch.long)
    return image, None, None, target_length

def predict_image(model, image_path):
    image, _, _, label_length = prepros_image(image_path)   
    # print('true:', main.decode_target(target))

    output = model(image.unsqueeze(0).cuda())
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    print('pred:', decode(output_argmax[0]))
    return

if __name__ == "__main__":
    model = torch.load('ctc3.pth')
    model.eval()
    dir_path = 'data'
    dir = os.listdir(dir_path)
    random.shuffle(dir)
    for file in dir[:10]:
        img_path = os.path.join(dir_path, file)
        # predict_image(model, img_path)
        predict_image(model, "/root/ZhihuScraper/captchas/labeled/3nBV_1622048517.png")