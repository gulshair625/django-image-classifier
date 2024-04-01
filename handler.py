import base64
import json
import logging
import traceback
from io import BytesIO
from sys import exc_info

import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
repo_name = "plsakr/vit-garbage-classification-v2"

label2id = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "organics": 3,
    "paper": 4,
    "plastic": 5,
    "trash": 6
}

id2label = {value: key for key, value in label2id.items()}
model = AutoModelForImageClassification.from_pretrained(
    repo_name,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
image_processor = AutoImageProcessor.from_pretrained(repo_name)


def predict(event, context):
    try:
        req_data = json.loads(event.get("body", json.dumps(event)))
        if 'image' not in req_data:
            return {'statusCode': 400,
                    "error": "No image provided"}

        img_data = base64.b64decode(req_data['image'])

        logger.info("Opening and processing the image...")

        image = image_processor(Image.open(BytesIO(img_data)).convert('RGB'), return_tensors="pt")

        with torch.no_grad():
            outputs = model(**image)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        if predicted_class == "trash":
            predicted_class = "organics"

        logger.info("Image processed successfully.")

        return {
            'statusCode': 200,
            'body': {
                "class": predicted_class.capitalize(),
                "probability": round(probabilities[0, predicted_class_idx].item() * 100)
            }
        }

    except AssertionError:
        logger.error(traceback.format_exc())
        return {
            'statusCode': 400,
            'body': {
                'error': f"{exc_info()[1]}"
            }
        }

    except:
        logger.error(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': {
                'error': f"{str(exc_info()[0]).split('class')[1].split('>')[0].strip()}: {exc_info()[1]}"
            }
        }
