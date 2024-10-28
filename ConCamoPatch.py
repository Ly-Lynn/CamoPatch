from torchvision import transforms
from ImageNetModels import ImageNetModel
from LossFunctions import UnTargeted, FaceVerification
import numpy as np
import argparse
import os
from Face_Recognition_Resource.get_architech import get_model
from time import time

from torchvision import transforms
from PIL import *


def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


from CamoPatch import Attack
if __name__ == "__main__":

    load_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="0 or 1", type=int, default=0)
    parser.add_argument("--model_name", type=str, help="restnet_vggface or restnet_webface", default="restnet_vggface")
    parser.add_argument("--attack", type=str, help="Type of attack: imagenet or face_ver", default="face_ver")

    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--temp", type=float, default=300.)
    parser.add_argument("--mut", type=float, default=0.3)
    parser.add_argument("--s", type=int, default=40)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--li", type=int, default=4)

    parser.add_argument("--image1_dir", type=str, help="Image1 File directory")
    parser.add_argument("--image2_dir", type=str, help="Image2 File directory if attack is face_ver", default=None)
    parser.add_argument("--true_label", type=int, help="Number of the correct label")
    parser.add_argument("--save_directory", type=str, help="Where to store the .npy files with the results")
    args = parser.parse_args()

    if args.attack == "face_ver":
        model = get_model(args.model_name)
        img1_dir = args.image1_dir
        img2_dir = args.image2_dir
        x_test1 = load_image(Image.open(img1_dir))
        x_test2 = load_image(Image.open(img2_dir))
        loss_func = FaceVerification(model, args.true_label, to_pytorch=True)
        x1 = pytorch_switch(x_test1).detach().numpy()
        x2 = pytorch_switch(x_test2).detach().numpy()
        params = {
            "x1": x1,
            "x2": x2,
            "eps": args.s**2,
            "n_queries": args.queries,
            "save_directory": args.save_directory + ".npy",
            "c": x1.shape[2],
            "h": x1.shape[0],
            "w": x1.shape[1],
            "N": args.N,
            "update_loc_period": args.li,
            "mut": args.mut,
            "temp": args.temp
        }
        attack = Attack(params, type_attack="face")
        attack.optimise(loss_func)

    else:
        model = ImageNetModel(args.model)
        image_dir = args.image_dir
        x_test = load_image(Image.open(image_dir))

        loss = UnTargeted(model, args.true_label, to_pytorch=True)
        x = pytorch_switch(x_test).detach().numpy()
        params = {
            "x": x,
            "eps": args.s**2,
            "n_queries": args.queries,
            "save_directory": args.save_directory + ".npy",
            "c": x.shape[2],
            "h": x.shape[0],
            "w": x.shape[1],
            "N": args.N,
            "update_loc_period": args.li,
            "mut": args.mut,
            "temp": args.temp
        }
        attack = Attack(params)
        attack.optimise(loss)
