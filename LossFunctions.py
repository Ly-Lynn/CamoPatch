import numpy as np
import torch
import math
import os
from .Face_Recognition_Resource.evalutation import calculate_similarity
from .Face_Recognition_Resource.utils import accuracy_FR

def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


def to_pytorch(tensor_image):
    return torch.from_numpy(tensor_image).permute(2, 0, 1)


class UnTargeted:
    def __init__(self, model, true, unormalize=False, to_pytorch=False):
        self.model = model
        self.true = true
        self.unormalize = unormalize
        self.to_pytorch = to_pytorch

    def get_label(self, img):
        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        return y

    def __call__(self, img):

        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
            preds = preds.tolist()
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        is_adversarial = True if y != self.true else False

        f_true = math.log(math.exp(preds[self.true]) + 1e-30)
        preds[self.true] = -math.inf

        f_other = math.log(math.exp(max(preds)) + 1e-30)
        return [is_adversarial, float(f_true - f_other)]


class Targeted:
    def __init__(self, model, true, target, unormalize=False, to_pytorch=False):
        self.model = model
        self.true = true
        self.target = target
        self.unormalize = unormalize
        self.to_pytorch = to_pytorch

    def get_label(self, img):
        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        return y

    def __call__(self, img):

        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
            preds = preds.tolist()
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        is_adversarial = True if y == self.target else False
        #print("current label %d target label %d" % (y, self.target))
        #print(self.target, len(preds))
        f_target = preds[self.target]
        #preds[self.true] = -math.inf

        f_other = math.log(sum(math.exp(pi) for pi in preds))
        return [is_adversarial, f_other - f_target]

class FaceVerification:
    def __init__(self, 
                 model, 
                 true,
                 unormalize=False):
        self.model = model
        self.true = true
        self.unormalize = unormalize

    def get_pred (self, img1, img2):
        if self.unormalize:
            img1_ = img1 * 255.

        else:
            img1_ = img1

        img1_ = to_pytorch(img1_)
        img1_ = img1_[None, :]
        img2_ = to_pytorch(img2)
        img2_ = img2_[None, :]
        
        preds1 = self.model(img1_)
        preds2 = self.model(img2_)
        sims = calculate_similarity(preds1, preds2)
        y = accuracy_FR(sims, self.true)
    
        return y, sims, 1-sims

    def __call__(self, img1, img2):
        y, sims, not_sims = self.get_pred(img1, img2)
        is_adversarial = True if y != self.true else False
        # label la 0 => not_sims > sims
        # label la 1 => sims > not_sims
        if self.true == 1:
            f_true = sims
            f_other = not_sims
        else:
            f_true = not_sims
            f_other = sims

        return [is_adversarial, float(f_true - f_other)]

