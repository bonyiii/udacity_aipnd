import argparse
import sys
from time import time, sleep
import os

import json

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def crop_center(im, new_width, new_height):
    width, height = im.size

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return im.crop((left, top, right, bottom))

def process_image(path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(path)
    image.thumbnail((256,256), Image.ANTIALIAS)
    image = crop_center(image, 224, 224)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    np_image = np.array(image, dtype = np.float64)
    np_image = np_image / 255.0
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2,0,1))
    return np_image

def load_model_from_checkpoint(path):
    checkpoint = torch.load(path + '/checkpoint.pth')
    model = getattr(models, checkpoint['arch'])(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()

    return model


def predict(model, image_path, gpu, top_k):
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    model = model.to(device)
    img = torch.FloatTensor([process_image(image_path)])
    img = img.to(device)

    with torch.no_grad():
        result = model(img)

    probs = np.array(torch.exp(result).topk(top_k)[0])
    classes = np.array(torch.exp(result).topk(top_k)[1])
    classes = [idx_to_class[c] for c in classes[0,:]]

    return(probs, classes)

def print_result(probs, classes, category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    for counter, klass in enumerate(classes):
        print(probs[0][counter])
        print(klass)


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', '-s', type = str, default = 'checkpoints', help = 'Checkopints directory name, Default: chekcpoints')
    parser.add_argument('--image',  '-d', type = str, help = 'Image to specify')
    parser.add_argument('--category_names', type = str, help = 'Mapping for category names')
    parser.add_argument('--top_k', type = int, default = 5, help = 'How many classes should be displayed')
    parser.add_argument('--gpu', '-g', type = bool, default = False, help = 'Whether or not use GPU acceleration, Default: False')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def main():
  start_time = time()
  # Creates & retrieves Command Line Arugments
  args = get_input_args()

  model = load_model_from_checkpoint(args.save_dir)
  probs, classes = predict(model, args.image, args.gpu, args.top_k)
  print_result(probs, classes, args.category_names)

  # Measure total program runtime by collecting end time
  end_time = time()

  # Computes overall runtime in seconds & prints it in hh:mm:ss format
  tot_time = end_time - start_time
  print("\n** Total Elapsed Runtime:",
        str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
        +str(int((tot_time%3600)%60)) )

# Call to main function to run the program
if __name__ == "__main__":
    main()
