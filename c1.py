import torch
import supervision
import transformers
import pytorch_lightning
import os
import torchvision
from transformers import DetrFeatureExtractor, DetrForObjectDetection

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

from transformers import DetrImageProcessor

image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
dataset = '../train_data/images'

ANNOTATION_FILE_NAME = "annotations.json"
TRAIN_DIRECTORY = os.path.join(dataset, "train")
VAL_DIRECTORY = os.path.join(dataset, "val")
TEST_DIRECTORY = os.path.join(dataset, "test")


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, 
        image_directory_path: str, 
        image_processor, 
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor, train=True)
VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)
TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)

print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))
print("Number of test examples:", len(TEST_DATASET))




import random
import cv2
import numpy as np
import supervision as sv

# select random image
image_ids = TRAIN_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
print('Image #{}'.format(image_id))

# load image and annotatons 
image = TRAIN_DATASET.coco.loadImgs(image_id)[0]
annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TRAIN_DATASET.root, image['file_name'])
image = cv2.imread(image_path)

# annotate
detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)

# we will use id2label function for training
categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}

labels = [
    f"{id2label[class_id]}" 
    for _, _, class_id, _ 
    in detections
]

box_annotator = sv.BoxAnnotator()
frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

 
sv.show_frame_in_notebook(image, (8, 8))



from torch.utils.data import DataLoader

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)







import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor


# settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
model.to(DEVICE)

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT, 
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
            
        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER




model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

batch = next(iter(TRAIN_DATALOADER))
outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])



from pytorch_lightning import Trainer

# settings
MAX_EPOCHS = 200

trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

trainer.fit(model)

MODEL_PATH = 'custom-model_200'
model.model.save_pretrained(MODEL_PATH)

# loading model
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(DEVICE)





















# import torch
# import supervision
# import transformers
# from transformers import DetrFeatureExtractor, DetrForObjectDetection

# feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
# from transformers import DetrImageProcessor

# image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
# import os
# import torchvision

# dataset = 'train_data/images'

# ANNOTATION_FILE_NAME = "annotations.json"
# TRAIN_DIRECTORY = os.path.join(dataset, "train")
# VAL_DIRECTORY = os.path.join(dataset, "val")
# TEST_DIRECTORY = os.path.join(dataset, "test")
# len1=0
# len2=0
# len3=0
# for i in os.listdir(TRAIN_DIRECTORY):
#   len1=len1+1
# print(len1)
# for i in os.listdir(TEST_DIRECTORY):
#   len2=len2+1
# print(len2)
# for i in os.listdir(VAL_DIRECTORY):
#   len3=len3+1
# print(len3)
# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(
#         self,
#         image_directory_path: str,
#         image_processor,
#         train: bool = True
#     ):
#         annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
#         super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
#         self.image_processor = image_processor

#     def __getitem__(self, idx):
#         images, annotations = super(CocoDetection, self).__getitem__(idx)
#         image_id = self.ids[idx]
#         annotations = {'image_id': image_id, 'annotations': annotations}
#         encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
#         pixel_values = encoding["pixel_values"].squeeze()
#         target = encoding["labels"][0]

#         return pixel_values, target


# TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor, train=True)
# VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)
# TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)
# print("Number of training examples:", len(TRAIN_DATASET))
# print("Number of validation examples:", len(VAL_DATASET))
# print("Number of test examples:", len(TEST_DATASET))

# import random
# import cv2
# import numpy as np
# import supervision as sv



# # select random image
# image_ids = TRAIN_DATASET.coco.getImgIds()
# image_id = random.choice(image_ids)
# print('Image #{}'.format(image_id))

# # load image and annotatons
# image = TRAIN_DATASET.coco.loadImgs(image_id)[0]
# annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
# image_path = os.path.join(TRAIN_DATASET.root, image['file_name'])
# image = cv2.imread(image_path)

# # annotate
# detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)

# # we will use id2label function for training
# categories = TRAIN_DATASET.coco.cats
# id2label = {k: v['name'] for k,v in categories.items()}

# labels = [
#     f"{id2label[class_id]}"
#     for _, _, class_id, _
#     in detections
# ]
# box_annotator = sv.BoxAnnotator()
# frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

# %matplotlib inline
# sv.show_frame_in_notebook(image, (8, 8))



# 11)from torch.utils.data import DataLoader

# def collate_fn(batch):
#     pixel_values = [item[0] for item in batch]
#     encoding = image_processor.pad(pixel_values, return_tensors="pt")
#     labels = [item[1] for item in batch]
#     return {
#         'pixel_values': encoding['pixel_values'],
#         'pixel_mask': encoding['pixel_mask'],
#         'labels': labels
#     }

# TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)
# VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)
# TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)
# import torch
# from transformers import DetrForObjectDetection, DetrImageProcessor


# # settings
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# CHECKPOINT = 'facebook/detr-resnet-50'
# CONFIDENCE_TRESHOLD = 0.5
# IOU_TRESHOLD = 0.8

# image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
# model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
# model.to(DEVICE)


# model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

# batch = next(iter(TRAIN_DATALOADER))
# outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

# import pytorch_lightning as pl
# from transformers import DetrForObjectDetection
# import torch

# class Detr(pl.LightningModule):

#     def __init__(self, lr, lr_backbone, weight_decay):
#         super().__init__()
#         self.model = DetrForObjectDetection.from_pretrained(
#             pretrained_model_name_or_path=CHECKPOINT,
#             num_labels=len(id2label),
#             ignore_mismatched_sizes=True
#         )

#         self.lr = lr
#         self.lr_backbone = lr_backbone
#         self.weight_decay = weight_decay

#     def forward(self, pixel_values, pixel_mask):
#         return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

#     def common_step(self, batch, batch_idx):
#         pixel_values = batch["pixel_values"]
#         pixel_mask = batch["pixel_mask"]
#         labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

#         outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

#         loss = outputs.loss
#         loss_dict = outputs.loss_dict

#         return loss, loss_dict

#     def training_step(self, batch, batch_idx):
#         loss, loss_dict = self.common_step(batch, batch_idx)
#         # logs metrics for each training_step, and the average across the epoch
#         self.log("training_loss", loss)
#         for k,v in loss_dict.items():
#             self.log("train_" + k, v.item())

#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, loss_dict = self.common_step(batch, batch_idx)
#         self.log("validation/loss", loss)
#         for k, v in loss_dict.items():
#             self.log("validation_" + k, v.item())

#         return loss

#     def configure_optimizers(self):
#         # DETR authors decided to use different learning rate for backbone
#         # you can learn more about it here:
#         # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
#         # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
#         param_dicts = [
#             {
#                 "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
#             {
#                 "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
#                 "lr": self.lr_backbone,
#             },
#         ]
#         return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

#     def train_dataloader(self):
#         return TRAIN_DATALOADER

#     def val_dataloader(self):
#         return VAL_DATALOADER



# from pytorch_lightning import Trainer

# # settings
# MAX_EPOCHS = 100

# trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

# trainer.fit(model)

# MODEL_PATH = 'custom-model'
# model.model.save_pretrained(MODEL_PATH)

# # loading model
# model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
# model.to(DEVICE)


