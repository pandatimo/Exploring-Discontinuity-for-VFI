import os
import torch
import argparse
from PIL import Image
from torchvision import transforms

from metric import *
import warnings
warnings.filterwarnings(action='ignore')


parser = argparse.ArgumentParser(description='VFI Test')
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--model', type=str, choices=['AdaCoF', 'CAIN', 'VFIT'])
parser.add_argument('--loss', type=bool, required=True)

class Test:
    def __init__(self, input_dir, resize):
        self.input_dir = input_dir
        if self.input_dir.split('/')[-1] == 'vimeo_septuplet':
            test_list = os.path.join(self.input_dir, 'sep_testlist.txt')
            with open(test_list, 'r') as f:
                self.im_list = f.read().splitlines()
        else:
            self.im_list = sorted(os.listdir(self.input_dir))
        self.resize = resize
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        test_name = self.input_dir.split('/')[-1]
        print(f'Test : {test_name}')
       
    @torch.no_grad()
    def Test(self, model):
        model.eval()
        metric_list = "PSNR-SSIM-LPIPS"
        metric = MetricCenter(metric_list, torch.device('cuda'))
        for idx in range(len(self.im_list)):
            if self.input_dir.split('/')[-1] == 'vimeo_septuplet':
                img0 = self.transform(Image.open(f'{self.input_dir}/sequences/{self.im_list[idx]}/im1.png')).unsqueeze(0).cuda()
                img1 = self.transform(Image.open(f'{self.input_dir}/sequences/{self.im_list[idx]}/im3.png')).unsqueeze(0).cuda()
                img2 = self.transform(Image.open(f'{self.input_dir}/sequences/{self.im_list[idx]}/im5.png')).unsqueeze(0).cuda()
                img3 = self.transform(Image.open(f'{self.input_dir}/sequences/{self.im_list[idx]}/im7.png')).unsqueeze(0).cuda()
                gt = self.transform(Image.open(f'{self.input_dir}/sequences/{self.im_list[idx]}/im4.png')).unsqueeze(0).cuda()
                
                img = []
                img.append(img0)
                img.append(img1)
                img.append(img2)
                img.append(img3)
                frame_out = model(img)
                metric.forward(frame_out, gt.cuda())

            else:
                file_len = len(os.listdir(self.input_dir + '/' + self.im_list[idx]))
                if file_len % 2 == 0:
                    file_len -= 1
                
                for f in range(3, file_len - 2, 2):
                    if self.resize == 1:
                        try:
                            img0 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f - 3).zfill(5)}.jpg')).unsqueeze(0).cuda()
                            img1 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f - 1).zfill(5)}.jpg')).unsqueeze(0).cuda()
                            img2 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f + 1).zfill(5)}.jpg')).unsqueeze(0).cuda()
                            img3 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f + 3).zfill(5)}.jpg')).unsqueeze(0).cuda()
                            gt = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f).zfill(5)}.jpg')).unsqueeze(0).cuda()

                        except:
                            img0 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f - 3).zfill(5)}.png')).unsqueeze(0).cuda()
                            img1 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f - 1).zfill(5)}.png')).unsqueeze(0).cuda()
                            img2 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f + 1).zfill(5)}.png')).unsqueeze(0).cuda()
                            img3 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f + 3).zfill(5)}.png')).unsqueeze(0).cuda()
                            gt = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f).zfill(5)}.png')).unsqueeze(0).cuda()

                    else:
                        try:
                            size = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f).zfill(5)}.jpg')).unsqueeze(0).size()
                            img0 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f - 3).zfill(5)}.jpg').resize((int(size[3]/self.resize), int(size[2]/self.resize)), Image.BILINEAR)).unsqueeze(0).cuda()
                            img1 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f - 1).zfill(5)}.jpg').resize((int(size[3]/self.resize), int(size[2]/self.resize)), Image.BILINEAR)).unsqueeze(0).cuda()
                            img2 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f + 1).zfill(5)}.jpg').resize((int(size[3]/self.resize), int(size[2]/self.resize)), Image.BILINEAR)).unsqueeze(0).cuda()
                            img3 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f + 3).zfill(5)}.jpg').resize((int(size[3]/self.resize), int(size[2]/self.resize)), Image.BILINEAR)).unsqueeze(0).cuda()
                            gt = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f).zfill(5)}.jpg').resize((int(size[3]/self.resize), int(size[2]/self.resize)), Image.BILINEAR)).unsqueeze(0).cuda()

                        except:
                            size = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f).zfill(5)}.png')).size()
                            img0 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f - 3).zfill(5)}.png').resize((int(size[3]/self.resize), int(size[2]/self.resize)), Image.BILINEAR)).unsqueeze(0).cuda()
                            img1 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f - 1).zfill(5)}.png').resize((int(size[3]/self.resize), int(size[2]/self.resize)), Image.BILINEAR)).unsqueeze(0).cuda()
                            img2 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f + 1).zfill(5)}.png').resize((int(size[3]/self.resize), int(size[2]/self.resize)), Image.BILINEAR)).unsqueeze(0).cuda()
                            img3 = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f + 3).zfill(5)}.png').resize((int(size[3]/self.resize), int(size[2]/self.resize)), Image.BILINEAR)).unsqueeze(0).cuda()
                            gt = self.transform(Image.open(f'{self.input_dir}/{self.im_list[idx]}/{str(f).zfill(5)}.png').resize((int(size[3]/self.resize), int(size[2]/self.resize)), Image.BILINEAR)).unsqueeze(0).cuda()

                    img = []
                    img.append(img0)
                    img.append(img1)
                    img.append(img2)
                    img.append(img3)
                    frame_out = model(img)

                    metric.forward(frame_out, gt.cuda())
                    
        metric_list = metric_list.split('-')
        for i in range(metric_list.__len__()):
            print(f'{metric_list[i]}: {metric.get_result()[i]}')

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.model == 'AdaCoF':
        from model.AdaCoF.adacofnet import AdaCoFNet
        model = AdaCoFNet().cuda()
        if args.loss:
            checkpoint = torch.load('./checkpoints/AdaCoF_L1.pth', map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load('./checkpoints/AdaCoF.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    
    elif args.model == 'CAIN':
        from model.CAIN.cain import CAIN
        model = CAIN(depth=3).cuda()
        if args.loss:
            checkpoint = torch.load('./checkpoints/CAIN_L1.pth', map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load('./checkpoints/CAIN.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

    elif args.model == 'VFIT':
        from model.VFIT.VFIT_B import UNet_3D_3D
        model = UNet_3D_3D(n_inputs=4, joinType="concat")
        model = torch.nn.DataParallel(model).cuda()
        if args.loss:
            checkpoint = torch.load('./checkpoints/VFIT_L1.pth', map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load('./checkpoints/VFIT.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

    Vimeo_test = Test('../DB/vimeo_septuplet', resize=1)
    Vimeo_test.Test(model)

    # DAVIS_test = Test('../DB/DAVIS', resize=2)
    # DAVIS_test.Test(model)

    # GDM_test = Test('../DB/GDM', resize=2)
    # GDM_test.Test(model)

if __name__ == "__main__":
    main()