from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import cv2
from torchvision import transforms
import torch
import random
import time

class SAMDataset(Dataset):
    def __init__(self, img_path, flow_path, processor, weight_path=None, crop_size=256):
        self.imgs = np.load(img_path)
        self.flows = np.load(flow_path)
        if weight_path is not None:
            self.weights = np.load(weight_path)
        else:
            self.weights = None

        self.processor = processor
        self.crop_size = crop_size

        self.isTrain = 'train' in img_path or 'train' in flow_path

    def __len__(self):
        return len(self.imgs)
    
    def _preprocess(self, img):

        #convert to grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #adaptive hist normalization
        # img_norm = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)

        #grab 2nd derivative via laplacian
        # edges = cv2.Laplacian(img_norm, cv2.CV_64F, ksize=3)
        # edges = img_norm

        #grab 1st derivative via sobel
        # edges = cv2.Sobel(img_norm, cv2.CV_64F, 1, 1, ksize=3)
        
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        #cvt to 3 channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img
    
    def _data_augmentation(self, img, label, weight):
        # ------------------------------------------------------------------------
        # 1) RANDOM FLIP (horizontal)
        # If we do a horizontal flip on the flow:
        #    - x-coordinates get mirrored, so dx -> -dx
        #    - y-coordinates do not change sign (dy -> dy)
        # ------------------------------------------------------------------------
        if random.random() > 0.5:
            # flip image horizontally
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)
            # if label has shape (3, H, W): (dy, dx, cell_prob)
            # we flip horizontally => x-axis reversed => dx -> -dx
            dx = label[0]
            dy = label[1]
            dx = -dx  # flip sign
            label[0] = dx
            label[1] = dy

        if weight is not None:
            weight = cv2.flip(weight, 1)

        # ------------------------------------------------------------------------
        # 2) RANDOM ROTATION + SCALE
        # We will rotate the image and each label channel as a normal image with cv2.warpAffine.
        # Then, to correct the *direction* of (dy, dx), we do a rotation of the vector.
        #
        # The rotation matrix for an angle (theta) is:
        #   [ cos(theta) -sin(theta)]
        #   [ sin(theta)  cos(theta)]
        #
        # But if the image is rotated by +theta,
        # the flow vectors effectively need to be rotated by -theta (opposite sign).
        # ------------------------------------------------------------------------
        angle_deg = random.randint(-180, 180)
        angle_rad = np.deg2rad(angle_deg)
        cosA = np.cos(-angle_rad)  # note the minus sign
        sinA = np.sin(-angle_rad)  # note the minus sign
        scale = random.uniform(0.8, 1.2)

        center = (img.shape[1] / 2, img.shape[0] / 2)  # (x, y)
        M = cv2.getRotationMatrix2D(center, angle_deg, scale)

        # Rotate + scale the image
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # label: shape (3, H, W) => we warp each channel as an image
        # AFTER that, we rotate the (dy, dx) vectors by -angle
        # (1) warp the channels
        dx_warped = cv2.warpAffine(label[0], M, (label.shape[2], label.shape[1]), flags=cv2.INTER_CUBIC)
        dy_warped = cv2.warpAffine(label[1], M, (label.shape[2], label.shape[1]), flags=cv2.INTER_CUBIC)
        cp_warped = cv2.warpAffine(label[2], M, (label.shape[2], label.shape[1]), flags=cv2.INTER_CUBIC)

        # (2) now rotate the flow vectors by -angle
        # old flow = (dy_warped, dx_warped)
        # new flow = R(-theta) * (dy, dx)
        # Typically we treat flows as (dy, dx) in "y,x" order,
        # so we apply:
        #    dy_new = dy*cos(-θ) - dx*sin(-θ)
        #    dx_new = dy*sin(-θ) + dx*cos(-θ)
        # Because we used cosA=cos(-θ) & sinA=sin(-θ), effectively:
        #    dy_new = dy_warped*cosA - dx_warped*sinA
        #    dx_new = dy_warped*sinA + dx_warped*cosA
        dx_final = dy_warped * sinA + dx_warped * cosA
        dy_final = dy_warped * cosA - dx_warped * sinA


        # put them back in label
        label = np.stack([dx_final, dy_final, cp_warped], axis=0)

        if weight is not None:
            weight = cv2.warpAffine(weight, M, (weight.shape[1], weight.shape[0]), flags=cv2.INTER_CUBIC)

        # prevent overflow
        img = img.astype(np.float32)

        #random brightness
        alpha = random.uniform(0.95, 1.05)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

        # #random contrast / gamma
        # beta = random.randint(-1, 1)
        # gamma = random.uniform(0.9, 1.1)
        # img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
        # img = np.power(img, gamma)

        #back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

        #randomly invert image
        if random.random() > 0.5:
            img = cv2.bitwise_not(img)

        return img, label, weight

    def __getitem__(self, idx):
        img = self.imgs[idx]
        image_orig = img.copy()
        label = self.flows[idx]
        if self.weights is not None:
            weight = self.weights[idx]
        else:
            weight = None

        #preprocess image
        img = self._preprocess(img)

        if self.isTrain:
            #data augmentation
            img, label, weight = self._data_augmentation(img, label, weight)

        #random crop to size
        x = random.randint(0, img.shape[1] - self.crop_size)
        y = random.randint(0, img.shape[0] - self.crop_size)
        img = img[y:y+self.crop_size, x:x+self.crop_size]
        label = label[y:y+self.crop_size, x:x+self.crop_size]
        if self.weights is not None:
            weight = weight[y:y+self.crop_size, x:x+self.crop_size]

        # return image_orig, img, label, weight #just for debugging

        # prepare image and prompt for the model
        inputs = self.processor(img, return_tensors="pt") #input image: shape: (256, 256, 3), range [0, 255]

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        inputs['ground_truth_mask'] = label
        # inputs['binary_mask'] = label > 0.001

        if self.weights is not None:
            inputs['weight'] = torch.tensor(weight).float()

        return inputs