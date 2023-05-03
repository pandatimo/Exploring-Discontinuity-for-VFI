import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from glob import glob

# libraries for Figure-Text Mixing
import string
import random
from tkinter import W
from PIL import Image, ImageDraw, ImageFont


def cointoss(p):
    return random.random() < p


class VimeoSepTuplet(Dataset):
    def __init__(self, data_root, is_training , input_frames=4, mode='mini'):
        """
        Dataloader with FTM data augmentation for Video frame interpolation.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames(2 / 4): # of frames to input for frame interpolation network.
            mode('mini' / 'full'): small version or full version dataset for evaluation
        """
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.training = is_training
        self.inputs = input_frames

        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        if mode != 'full':
            tmp = []
            for i, value in enumerate(self.testlist):
                if i % 38 == 0:
                    tmp.append(value)
            self.testlist = tmp

        self.transforms = transforms.ToTensor()

        # augment list
        self.random_crop = (256, 256)
        self.augment_flip = True
        self.augment_TM = True
        self.augment_FM = True
        self.augment_t = True
        
        if self.augment_TM:
            # Components of striing contain letters, digits and space
            self.string_letter = string.ascii_letters + string.digits + " "
            # We have used the default fonts provided by Windows
            self.font_list = glob('../DB/font/*.ttf') + glob('../DB/font/*.TTF')

    def Figure_Mixing(self, imgs, h, w):
        """
        Figure Mixing Data Augmentation
        imgs : numpy array list
        """
        if cointoss(0.5):
            r = random.randrange(5, 21)
            x, y = random.randrange(0, h - r), random.randrange(0, w - r)
            rgb = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))
            thickness = random.randrange(1, 4)
            for idx, _ in enumerate(imgs):
                cv2.circle(imgs[idx], (x, y), r, rgb, thickness)

        if cointoss(0.5):
            dx, dy = random.randrange(10, 41), random.randrange(10, 41)
            x, y = random.randrange(0, h - dx), random.randrange(0, w - dy)
            rgb = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))
            thickness = random.randrange(1, 4)
            for idx, _ in enumerate(imgs):
                cv2.rectangle(imgs[idx], (x, y), (x + dx, y + dy), rgb, thickness)

        return imgs

    def Text_Mixing(self, imgs, h, w):
        """
        Text Mixing Data Augmentation
        imgs : Image list
        """
        if cointoss(0.5):
            # setting parameters for TM
            letter = ""
            for i in range(random.randrange(5, 30)):
                letter += random.choice(self.string_letter)

            selectedFont = ImageFont.truetype(os.path.join(random.choice(self.font_list)), random.randrange(10, 40))
            text_rgb = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))
            x, y = random.randrange(0, h), random.randrange(0, w)
            stroke_rgb = (random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256))
            stroke_width = random.randrange(0, 3)

            # determine shifting text or not
            if cointoss(0.5):
                rand = random.random() * 3
                dy = random.randrange(-40, -20)
                if rand < 1:
                    dy = 0
                elif rand < 2:
                    dy = -dy
                else:
                    dy = dy

                if self.input == 2:
                    draw = ImageDraw.Draw(imgs[0])
                    draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                    draw = ImageDraw.Draw(imgs[1])
                    draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                    draw = ImageDraw.Draw(imgs[2])
                    draw.text((x, y + dy), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                else:
                    draw = ImageDraw.Draw(imgs[0])
                    draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                    draw = ImageDraw.Draw(imgs[1])
                    draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                    draw = ImageDraw.Draw(imgs[2])
                    draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                    draw = ImageDraw.Draw(imgs[3])
                    draw.text((x, y + dy), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                    draw = ImageDraw.Draw(imgs[4])
                    draw.text((x, y + dy), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)

            else:
                if cointoss(0.5):
                    if self.input == 2:
                        draw = ImageDraw.Draw(imgs[0])
                        draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                        draw = ImageDraw.Draw(imgs[1])
                        draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                    else:
                        draw = ImageDraw.Draw(imgs[0])
                        draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                        draw = ImageDraw.Draw(imgs[1])
                        draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                        draw = ImageDraw.Draw(imgs[2])
                        draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                else:
                    if self.input == 2:
                        draw = ImageDraw.Draw(imgs[2])
                        draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                    else:
                        draw = ImageDraw.Draw(imgs[3])
                        draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)
                        draw = ImageDraw.Draw(imgs[4])
                        draw.text((x, y), letter, fill=text_rgb, font=selectedFont, stroke_width=stroke_width, stroke_fill=stroke_rgb)

        return imgs

    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])

        rawFrames = []
        if self.input == 2:
            rawFrames.append(Image.open(imgpath + '/im3.png'))
            rawFrames.append(Image.open(imgpath + '/im4.png'))
            rawFrames.append(Image.open(imgpath + '/im5.png'))

        elif self.input == 4:
            rawFrames.append(Image.open(imgpath + '/im1.png'))
            rawFrames.append(Image.open(imgpath + '/im3.png'))
            rawFrames.append(Image.open(imgpath + '/im4.png'))
            rawFrames.append(Image.open(imgpath + '/im5.png'))
            rawFrames.append(Image.open(imgpath + '/im7.png'))

        # Data augmentation
        if self.training:
            images = []
            if self.random_crop is not None:
                i, j, h, w = transforms.RandomCrop.get_params(rawFrames[0], output_size=self.random_crop)
                for idx, fr in enumerate(rawFrames):
                    rawFrames[idx] = TF.crop(fr, i, j, h, w)

            if self.augment_flip:
                if cointoss(0.5):
                    for idx, fr in enumerate(rawFrames):
                        rawFrames[idx] = TF.hflip(fr)

                if cointoss(0.5):
                    for idx, fr in enumerate(rawFrames):
                        rawFrames[idx] = TF.vflip(fr)

            for idx, fr in enumerate(rawFrames):
                rawFrames[idx] = self.transforms(fr)

            # applied FTM
            guide_map = [Image.fromarray(np.zeros((self.random_crop[0], self.random_crop[1], 3), np.uint8), 'RGB') for _ in range(self.input + 1)]
            if self.augment_TM:
                guide_map = self.Text_Mixing(guide_map, self.random_crop[0], self.random_crop[1])

            guide_map = [np.array(i, np.uint8) for i in guide_map]

            if self.augment_FM:
                guide_map = self.Figure_Mixing(guide_map, self.random_crop[0], self.random_crop[1])

            guide_map = [self.transforms(i) for i in guide_map]

            # Random Temporal Flip
            if self.augment_t:
                if cointoss(0.5):
                    for idx, fr in guide_map:
                        rawFrames[idx][fr > 0] = fr[fr > 0]
                        if idx != self.input // 2:
                            images.append(rawFrames[idx])

                else:
                    for idx, fr in guide_map:
                        rawFrames[self.input - idx][fr > 0] = fr[fr > 0]
                        if idx != self.input // 2:
                            images.append(rawFrames[self.input - idx])

            gt = rawFrames[self.input // 2]
            Dmap = torch.zeros(1, self.random_crop[0], self.random_crop[1])

            Dmap[guide_map[self.input // 2][0].unsqueeze(0) > 0] = 1
            Dmap[Dmap != 1] = 0

            return images, gt, Dmap
        else:
            images = []
            for idx, fr in enumerate(rawFrames):
                if idx != self.input // 2:
                    images.append(self.transforms(fr))
            gt = rawFrames[self.input // 2]
            imgpath = '_'.join(imgpath.split('/')[-2:])

            return images, gt

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)
          
