# backend/segmentation.py
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import json

#path = "../models/nvidia/segformer-b1-finetuned-ade-512-512/"

def load_config(config_path = "../config.json"):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

config = load_config()

feature_extractor_path = config["feature_extractor"]["feature_extractor_path"]
model_path = config["model"]["model_path"]

model = SegformerForSemanticSegmentation.from_pretrained(model_path)
feature_extractor = SegformerFeatureExtractor.from_pretrained(feature_extractor_path)
model.eval()




def random_colormap(num_classes):
    return np.random.randint(0, 255, size = (num_classes, 3), dtype = np.uint8)


COLORMAP = random_colormap(150)


def segment(image_array):

    img_arr = np.array(image_array)

    stg1 = feature_extractor(img_arr, return_tensors = 'pt')

    with torch.no_grad():
        outputs = model(**stg1)

    logits = outputs.logits
    segmentation_map = logits.argmax(dim = 1)[0].cpu().numpy()
    segmentation_color = COLORMAP[segmentation_map]

    segmentation_color = cv2.resize(segmentation_color, (img_arr.shape[1], img_arr.shape[0]))
    result_image = cv2.addWeighted(img_arr, 0.5, segmentation_color, 0.5, 0)

    return result_image