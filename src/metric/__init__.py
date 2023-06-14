import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy import linalg
from scipy.stats import entropy
from torch.autograd import Variable
from .lpipsmodules.lpips import LPIPS as Lpips
from .fidmodules.inception import InceptionV3
from torchvision.models.inception import inception_v3


# set like loss class
class MetricCenter():
    def __init__(self, opt, dev):
        self.metric_list = opt.split('-')
        self.metric_calculator_init(dev)

    def metric_calculator_init(self, dev):
        self.metric = []
        for metric in self.metric_list:
            if metric == 'PSNR':
                self.metric.append(PSNR())
            elif metric == 'SSIM':
                self.metric.append(SSIM())
            elif metric == 'LPIPS':
                self.metric.append(LPIPS(dev))
            elif metric == 'FID':
                self.metric.append(FID(dev))
            elif metric == 'IS':
                self.metric.append(InceptionScore(dev))
            else:
                raise Exception('unknown metric: {}'.format(metric))

    def forward(self, pred, gt):
        for metric in self.metric:
            metric.calculate(pred, gt)

    def prime_metric(self, pred, gt):
        self.metric[0].calculate(pred, gt)

    def get_metric_list(self):
        return self.metric_list

    def get_result(self, prime=False):
        metric_results = []
        if prime:
            metric_results.append(self.metric[0].get_result())
        else:
            for metric in self.metric:
                metric_results.append(metric.get_result())
        return metric_results


def best_metric(best, target, metric):
    if metric in ['PSNR', 'SSIM', 'IS']:
        if target >= best:
            return True
    elif metric in ['LPIPS', 'FID']:
        if target <= best:
            return True
    else:
        raise Exception('unknown metric: {}'.format(metric))

    return False


class PSNR():
    def __init__(self):
        self.values_bucket = []

    def calculate(self, output, target):
        for pred, gt in zip(output, target):
            pred = pred.clamp(0.0, 1.0)
            mse = torch.pow(gt - pred, 2).mean()
            if mse == 0:
                return float("inf")
            self.values_bucket.append(10 * torch.log10(1 / mse).item())

    def get_last_value(self):
        return self.values_bucket[-1]

    def get_result(self):
        return np.array(self.values_bucket).mean()


class SSIM():
    def __init__(self):
        self.values_bucket = []

    def calculate(self, output, target, window_size=11, size_average=True):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()

        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
            return window

        def _ssim(img1, img2, window, channel, size_average=True):
            mu1 = torch.nn.functional.conv2d(img1, window, groups=channel)
            mu2 = torch.nn.functional.conv2d(img2, window, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, groups=channel) - mu1_sq
            sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, groups=channel) - mu2_sq
            sigma12 = torch.nn.functional.conv2d(img1 * img2, window, groups=channel) - mu1_mu2

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            if size_average:
                return ssim_map.mean()
            else:
                return ssim_map.mean(1).mean(1).mean(1)

        for pred, gt in zip(output, target):
            img1 = pred.clamp(0.0, 1.0).unsqueeze(dim=0)
            img2 = gt.unsqueeze(dim=0)

            (_, channel, _, _) = img1.size()
            window = create_window(window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.values_bucket.append(_ssim(img1, img2, window, channel, size_average).item())

    def get_last_value(self):
        return self.values_bucket[-1]

    def get_result(self):
        return np.array(self.values_bucket).mean()


class LPIPS():
    def __init__(self, dev):
        self.criterion = Lpips().to(dev)
        self.values_bucket = []

    def calculate(self, output, target):
        output = output.clamp(0.0, 1.0) * 2.0 - 1.0
        target = target * 2.0 - 1.0

        for pred, gt in zip(output, target):
            self.values_bucket.append(self.criterion(pred.unsqueeze(dim=0), gt.unsqueeze(dim=0)).item())

    def get_last_value(self):
        return self.values_bucket[-1]

    def get_result(self):
        return np.array(self.values_bucket).mean()


class FID():
    def __init__(self, dev, FID_dims=2048):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_dims]
        self.model = InceptionV3([block_idx]).to(dev).eval()
        self.target_scores = []
        self.output_scores = []

    def get_activation(self, image):
        pred = self.model(image)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.cpu().data.numpy().reshape(pred.size(0), -1)

    @torch.no_grad()
    def calculate(self, output, target):
        for pred, gt in zip(output, target):
            self.target_scores.append(self.get_activation(gt.unsqueeze(dim=0)))
            self.output_scores.append(self.get_activation(pred.unsqueeze(dim=0)))

    def get_last_value(self):
        return None

    def get_result(self):
        target_activation = np.vstack(tuple(self.target_scores))    # size: (num of test images, FID_dims)
        output_activation = np.vstack(tuple(self.output_scores))

        mu_target = np.mean(target_activation, axis=0)
        std_target = np.cov(target_activation, rowvar=False)

        mu_output = np.mean(output_activation, axis=0)
        std_output = np.cov(output_activation, rowvar=False)

        return self.calculate_frechet_distance(mu_target, std_target, mu_output, std_output)

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


class InceptionScore(object):
    def __init__(self, dev):
        self.model = inception_v3(pretrained=True, transform_input=False).to(dev).eval()
        self.values_bucket = []

    def resize(self, x):
        x = F.interpolate(x, size=(299, 299))
        return x

    @torch.no_grad()
    def calculate(self, output, target):
        output = output.clamp(0.0, 1.0) * 2.0 - 1.0
        x = self.resize(output)
        x = self.model(x)
        x = F.softmax(x, dim=1).data.cpu()
        self.values_bucket.append(x)

    def get_result(self):
        preds = torch.cat(self.values_bucket, dim=0).numpy()
        N = preds.shape[0]

        split_scores = []
        for k in range(10):
            part = preds[k * (N//10): (k+1) * (N//10), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        mean = np.mean(split_scores)
        # std = np.std(split_scores)
        return mean


def best_metric_initialization(opt):
    metric = opt.metric.split('-')[0]

    if (metric == 'PSNR') or (metric == 'SSIM'):
        best_metric_init = 0
    elif (metric == 'LPIPS') or (metric == 'FID'):
        best_metric_init = 1e6
    else:
        raise Exception("metric {} is not defined".format(metric))

    return best_metric_init


