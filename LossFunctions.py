import numpy as np
import torch
import math
import os
import sys
current_dir  = os.getcwd()

resource_path = os.path.join(current_dir, 'Face_Recognition_Resource')
sys.path.append(resource_path)

from Face_Recognition_Resource.evalutation import calculate_similarity
from Face_Recognition_Resource.utils import get_predict

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
        
        preds1 = self.model(img1_.to('cuda'))
        preds2 = self.model(img2_.to('cuda'))
        sims = calculate_similarity(preds1, preds2)
        y = get_predict(sims)
    
        return sims

    def __call__(self, img1, img2):
        sims  = self.get_pred(img1, img2)
        # is_adversarial = True if y != self.true else False
        
        if isinstance(sims, torch.Tensor):
            adv_scores = (1 - self.true) * (0.5 - sims) + self.true * (sims - 0.5)
            adv_scores = float(adv_scores.cpu().item())
        else:
            adv_scores = (1 - self.true) * (0.5 - sims) + self.true * (sims - 0.5)
        is_adversarial = True if adv_scores > 0 else False
        # print(adv_scores, is_adversarial, sep='\n')
        return [is_adversarial, adv_scores]

