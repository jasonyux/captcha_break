import random, re
import torch, os, string
from PIL import Image
from torchvision.transforms.functional import to_tensor

LABEL_LENGTH = 4
FILE_LABEL_PATTERN = r'([0-9a-zA-Z]+)_([0-9]+)(\.png)'
CHARACTERS = string.digits + string.ascii_uppercase + string.ascii_lowercase
GPU_MODE = torch.cuda.is_available()

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
    if image.mode != 'RGB': # need 3 channel RGB
	    image = image.convert('RGB')
    image = to_tensor(image)
    target_length = torch.full(size=(1, ), fill_value=LABEL_LENGTH, dtype=torch.long)
    return image, None, None, target_length

def predict_image(model, image_path):
    image, _, _, label_length = prepros_image(image_path)
    if GPU_MODE==False:
        output = model(image.unsqueeze(0))
    else:
        output = model(image.unsqueeze(0).cuda())    
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    result = decode(output_argmax[0])
    print(f'pred:{result}')
    return result


"""
Expects label in of the image in the format specified in @FILE_LABEL_PATTERN
"""
def get_label(actual_img_path):
    file = os.path.split(actual_img_path)[-1]
    matchObj = re.match(FILE_LABEL_PATTERN, file)
    return matchObj.group(1)


def check_result(pred:str, actual_img_path):
    label = get_label(actual_img_path)
    print(f'actual={label}')
    return label==pred


if __name__ == "__main__":
    if GPU_MODE==False:
        model = torch.load('ctc3.pth', map_location=torch.device('cpu'))
    else:
        model = torch.load('ctc3.pth')
    model.eval()
    dir_path = '/root/ZhihuScraper/captchas/labeled'
    dir = os.listdir(dir_path)
    random.shuffle(dir)
    fully_correct = 0.0
    partial_correct = 0.0
    for file in dir[:]:
        img_path = os.path.join(dir_path, file)
        # predict_image(model, img_path)
        pred = predict_image(model, img_path)
        actual = get_label(img_path)
        if pred == actual:
            fully_correct += 1.
        if pred.lower() == actual.lower():
            partial_correct += 1.
    print(f"full success rate={fully_correct/len(dir)} partial success rate={partial_correct/len(dir)}")
