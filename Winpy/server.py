from flask import Flask, request
import os, torch
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision.transforms.functional import to_tensor
from recognize import decode

DOWNLOAD_PATH = "unlabeled"
GPU_MODE = torch.cuda.is_available()
LABEL_LENGTH = 4
MODEL_PATH = 'ctc3.pth'

app = Flask(__name__)

"""
assumes opened_img from Image.open('xxx')
"""
def predict_image(opened_img):
    if not hasattr(predict_image, 'model'):
        if GPU_MODE==False:
            predict_image.model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        else:
            predict_image.model = torch.load(MODEL_PATH)
        predict_image.model.eval()
    
    model = predict_image.model
    if opened_img.mode != 'RGB': # need 3 channel RGB
	    opened_img = opened_img.convert('RGB')
    image = to_tensor(opened_img)
    if GPU_MODE==False:
        output = model(image.unsqueeze(0))
    else:
        output = model(image.unsqueeze(0).cuda())    
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    result = decode(output_argmax[0])
    print(f'pred:{result}')
    return result

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/ocr', methods=['POST'])
def upload():
    file = request.files['file']
    # filename = secure_filename(file.filename)
    img = Image.open(file)
    print(img.size)
    return predict_image(img)