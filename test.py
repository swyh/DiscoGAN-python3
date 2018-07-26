import argparse
import torch
import os
import cv2
import numpy as np
from scipy import misc
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--epoch', type=str, default='-0.0', help='Set epoch')
parser.add_argument('--result_path', type=str, default='./result/', help='Set the path the result images will be saved.')
parser.add_argument('--input_path', type=str, default='./input/', help='Set the path the input images will be test.')
parser.add_argument('--model_path', type=str, default='./models/handbags2shoes/discogan', help='Set the path for trained models')
parser.add_argument('--type', type=str, default='A', help='Set type(A or B)')

global args
args = parser.parse_args()

epoch = args.epoch
result_path = args.result_path
input_path = args.input_path
model_path = args.model_path
type = args.type


def get_model():
    generator_A = torch.load(os.path.join(model_path, 'model_gen_A') + epoch)
    generator_B = torch.load(os.path.join(model_path, 'model_gen_B') + epoch)

    if torch.cuda:
        generator_A = generator_A.cuda()
        generator_B = generator_B.cuda()

    return generator_A, generator_B


def get_real_image(image_size=64): # path 불러오기
    images = []

    file_list = os.listdir(input_path)
    for file in file_list:
        if file.endswith(".jpg"):
            file_path = os.path.join(input_path, file)

            print(file_path)
            image = cv2.imread(file_path)  # fn이 한글 경로가 포함되어 있으면 제대로 읽지 못함. binary로 바꿔서 처리하는 방법있음

            if image is None:
                print("None")
                continue
            # image를 image_size(default=64)로 변환
            image = cv2.resize(image, (image_size, image_size))
            image = image.astype(np.float32) / 255.
            image = image.transpose(2, 0, 1)
            images.append(image)

    if images:
        print("push the stack")
        images = np.stack(images)
    else:
        print("error, images is emtpy")

    return images


def save_image(name, images):
    for i in range(0, len(images)):
        print("save image")
        image = images[i].cpu().data.numpy().transpose(1, 2, 0) * 255.
        misc.imsave(os.path.join(result_path, name + "_" + str(i) + '.jpg'), image.astype(np.uint8)[:, :, ::-1])


def main():
    generator_A, generator_B = get_model()
    images = get_real_image()

    A = Variable(torch.FloatTensor(images))

    if torch.cuda:
        A = A.cuda()

    if type == 'A':
        AB = generator_B(A)
        ABA = generator_A(AB)
        save_image("A", A)
        save_image("AB", AB)
        save_image("ABA", ABA)
    else:
        BA = generator_A(A)
        BAB = generator_B(BA)
        save_image("B", A)
        save_image("BA", BA)
        save_image("BAB", BAB)


if __name__=="__main__":
    main()
