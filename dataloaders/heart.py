from base import BaseDataSet, BaseDataLoader
from utils import pallete
from utils import validcut
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import json
import random
import SimpleITK as sitk
from matplotlib import pyplot as plt
from skimage import exposure

class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image):
        image = np.asarray(image)
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        image = transforms.functional.to_pil_image(image)
        return image
class PairHEARTDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 4
        self.datalist = 0
        self.stride = 8
        self.iou_bound = [0.3, 0.7]  # default [0.3, 0.7]
        self.resizer = transforms.Resize((512, 512))
        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(PairHEARTDataset, self).__init__(**kwargs)
        self.iscut = False
        self.ch = 4
        self.trainlist = [i for i in range(self.n_labeled_examples+1,len(os.listdir(self.root))+1)]+[i for i in range(450+self.n_labeled_examples+1,450+len(os.listdir(self.root))+1)]
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            RandomGaussianBlur(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),#改变图像亮度、对比度、饱和度、色调
            transforms.RandomGrayscale(p=1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.3),
            self.normalize,
        ])
    def __len__(self):
        if self.use_weak_lables == False:
            return len(os.listdir(self.root))*2
        else:
            return (len(os.listdir(self.root))-self.n_labeled_examples)*2
    def _set_files(self):
        return 0

    def cut(self, rlabel, image):
        contours, her = cv2.findContours(rlabel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(image.shape)
        bound = cv2.boundingRect(np.array(contours[0]))  # 左上角坐标 x y 矩形的宽，高 w，h
        h, w = image.shape
        x, y, w1, h1 = bound
        # print(x, y, w1, h1)
        x, y, w1, h1 = validcut.boderexpand(x, y, w1, h1, w, h,mode='train')
        # print(x, y, w1, h1)
        image[:y] = 0
        image[y + h1:] = 0
        for i in range(0, h):
            image[i][:x] = 0
            image[i][x + w1:] = 0
        # cv2.drawContours(image, contours, -1, (255), 2)
        image = image.reshape(1, h, w)
        return image
    def _load_data(self, index):
        index = index + 1
        sum = len(os.listdir(self.root))
        if (index <= sum):
            patient = 'patient' + format(index, '04d')
            image_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ED.mhd'
            label_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ED_gt.mhd'
        else:
            patient = 'patient' + format(index - sum, '04d')
            image_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ES.mhd'
            label_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ES_gt.mhd'
        if self.use_weak_lables == True:
            # print(self.trainlist)
            if index >(450 - self.n_labeled_examples):
                type = 'ES'
                patient = 'patient' + format(self.trainlist[index - 1]-450, '04d')
            else:
                type = 'ED'
                patient = 'patient' + format(self.trainlist[index - 1], '04d')
            label_path = '/mnt/data/guoziyu/pseudolabels/'+str(self.n_labeled_examples)+'/'+patient+'_2CH_'+type+'.png'
        image_rawfile = sitk.ReadImage(image_path)
        label_rawfile = sitk.ReadImage(label_path)
        image = sitk.GetArrayFromImage(image_rawfile)
        label = sitk.GetArrayFromImage(label_rawfile)
        # if self.use_weak_lables == True:
        #     label = label.reshape(3,512,512)
        # print(label.shape)
        # image = cv2.resize(image[0], (512, 512))
        # image = image.reshape(1, 512, 512)
        if self.iscut == True:
            image = self.cut(label[0],image[0])
        image = np.stack((image[0], image[0], image[0]),axis=2)
        # label = cv2.merge((label[0], label[0], label[0]))
        image = torchvision.transforms.ToPILImage()(image)
        label = torchvision.transforms.ToPILImage()(label[0])
        # # image = torchvision.transforms.ToPILImage()(image[0])
        # label = torchvision.transforms.ToPILImage()(label[0])
        image = self.resizer(image)
        label = self.resizer(label)
        image = self.to_tensor(image)
        label = self.to_tensor(label)
        # image = cv2.resize(image,(512, 512),interpolation=cv2.INTER_AREA)
        # print(label.shape)
        # label = cv2.resize(label[0],(512,512),interpolation=cv2.INTER_AREA)
        # print(label.shape)
        # image = image.reshape(512, 512, 3)
        label = label.reshape(512, 512, 1)
        # print(image.shape)
        image = image*255
        label = label*255
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.int32)
        image_id = index
        image = image.transpose(1,2,0)
        # print(image.shape)
        # print(label.shape)
        # image = image.transpose(1, 2, 0)
        # plt.subplot(121), plt.imshow(image, cmap="gray")
        # plt.subplot(122), plt.imshow(label, cmap="gray")
        # plt.show()
        return image, label, image_id

    def __getitem__(self, index):
        # image_path = os.path.join(self.root, self.files[index][1:])
        # image = np.asarray(Image.open(image_path))
        # if self.use_weak_lables:
        #     label_path = os.path.join(self.weak_labels_output, image_id + ".png")
        # else:
        #     label_path = os.path.join(self.root, self.labels[index][1:])
        # label = np.asarray(Image.open(label_path), dtype=np.int32)
        image,label,image_id = self._load_data(index)
        # print(image.shape)
        h, w, _ = image.shape
        # print(h)
        # print(w)
        # self.base_size = 400
        longside = random.randint(int(self.base_size * 0.8), int(self.base_size * 2.0)) #返回320-800之间的任意整数
        # h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
        # h,w = 512,512
        h, w = self.crop_size, self.crop_size
        image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        crop_h, crop_w = self.crop_size, self.crop_size  #self.crop_size = 320
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, }
        if pad_h > 0 or pad_w > 0:
            # image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs) #在图像外面绘制边界框
            image = cv2.copyMakeBorder(image, value=0, **pad_kwargs) #在图像外面绘制边界框
            label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)

        x1 = random.randint(0, w + pad_w - crop_w)
        y1 = random.randint(0, h + pad_h - crop_h)

        max_iters = 50
        k = 0
        while k < max_iters:
            x2 = random.randint(0, w + pad_w - crop_w)
            y2 = random.randint(0, h + pad_h - crop_h)
            # crop relative coordinates should be a multiple of 8  裁剪的相对坐标应该8的倍数
            x2 = (x2 - x1) // self.stride * self.stride + x1 #//表示整数的除法
            y2 = (y2 - y1) // self.stride * self.stride + y1
            if x2 < 0: x2 += self.stride
            if y2 < 0: y2 += self.stride

            if crop_w - abs(x2 - x1) < 0 or crop_h - abs(y2 - y1) < 0:
                k += 1
                continue

            inter = (crop_w - abs(x2 - x1)) * (crop_h - abs(y2 - y1))
            union = 2 * crop_w * crop_h - inter
            iou = inter / union
            if iou >= self.iou_bound[0] and iou <= self.iou_bound[1]:
                break
            k += 1

        if k == max_iters:
            x2 = x1
            y2 = y1

        overlap1_ul = [max(0, y2 - y1), max(0, x2 - x1)]  #左上角
        overlap1_br = [min(self.crop_size, self.crop_size + y2 - y1, h // self.stride * self.stride),
                       min(self.crop_size, self.crop_size + x2 - x1, w // self.stride * self.stride)] #右下角
        overlap2_ul = [max(0, y1 - y2), max(0, x1 - x2)]
        overlap2_br = [min(self.crop_size, self.crop_size + y1 - y2, h // self.stride * self.stride),
                       min(self.crop_size, self.crop_size + x1 - x2, w // self.stride * self.stride)]

        try:
            assert (overlap1_br[0] - overlap1_ul[0]) * (overlap1_br[1] - overlap1_ul[1]) == (
                        overlap2_br[0] - overlap2_ul[0]) * (overlap2_br[1] - overlap2_ul[1])
        except:
            print("h: {}, w: {}".format(h, w))
            print("image.shape: ", image.shape)
            print("x1: {}, x2: {}, y1: {}, y2: {}".format(x1, x2, y1, y2))
            # print("image_path:", image_path)
            print("ul1: ", overlap1_ul)
            print("br1: ", overlap1_br)
            print("ul2: ", overlap2_ul)
            print("br2: ", overlap2_br)
            print("index: ", index)
            exit()

        image1 = image[y1:y1 + self.crop_size, x1:x1 + self.crop_size].copy()
        image2 = image[y2:y2 + self.crop_size, x2:x2 + self.crop_size].copy()
        label1 = label[y1:y1 + self.crop_size, x1:x1 + self.crop_size].copy()
        label2 = label[y2:y2 + self.crop_size, x2:x2 + self.crop_size].copy()
        try:
            assert image1[overlap1_ul[0]:overlap1_br[0], overlap1_ul[1]:overlap1_br[1]].shape == image2[overlap2_ul[0]:
                                                                                                        overlap2_br[0],
                                                                                                 overlap2_ul[1]:
                                                                                                 overlap2_br[1]].shape
        except:
            print("h: {}, w: {}".format(h, w))
            print("image.shape: ", image.shape)
            print("x1: {}, x2: {}, y1: {}, y2: {}".format(x1, x2, y1, y2))
            # print("image_path:", image_path)
            print("ul1: ", overlap1_ul)
            print("br1: ", overlap1_br)
            print("ul2: ", overlap2_ul)
            print("br2: ", overlap2_br)
            print("index: ", index)
            exit()

        flip1 = False
        if random.random() < 0.5:
            image1 = np.fliplr(image1)
            label1 = np.fliplr(label1)
            flip1 = True

        flip2 = False
        if random.random() < 0.5:
            image2 = np.fliplr(image2)
            label2 = np.fliplr(label2)
            flip2 = True
        flip = [flip1, flip2]
        # w, h = label1.shape
        # ratio = random.randint(-15, 15)
        # M = cv2.getRotationMatrix2D((w // 2, h // 2), ratio, 1.0)
        # image1 = cv2.warpAffine(image1,M,(w,h))
        # label1 = cv2.warpAffine(label1.astype("uint8"),M,(w,h))
        # image2 = cv2.warpAffine(image2,M,(w,h))
        # label2 = cv2.warpAffine(label2.astype("uint8"),M,(w,h))
        # label1 = label1.astype("int32")
        # label2 = label2.astype("int32")
        image1 = self.train_transform(image1)
        image2 = self.train_transform(image2)
        images = torch.stack([image1, image2])
        labels = torch.from_numpy(np.stack([label1, label2]))
        # overlap1_ul = [0, 0]
        # overlap1_br = [320, 320]
        # overlap2_ul = [8, 8]
        # overlap2_br = [328, 328]
        # print(overlap1_ul)
        # print(overlap1_br)
        # print(overlap2_ul)
        # print(overlap2_br)
        # print(label1.shape)
        # plt.subplot(121),plt.imshow(image1[0].reshape(320,320,1),cmap="gray")
        # print(image1.shape)
        # print(label1.shape)
        # print(image.shape)
        # print(label2.shape)
        # print(image.shape)
        transforms.ToPILImage()(image1).save("image1.png",'PNG')
        transforms.ToPILImage()(image2).save("image2.png",'PNG')
        # plt.subplot(131),plt.imshow(image, cmap="gray")
        # plt.subplot(132), plt.imshow(image1[0], cmap="gray")
        # plt.subplot(133), plt.imshow(image2[0], cmap="gray")
        # plt.title(str(index))
        # plt.show()


        return images, labels, overlap1_ul, overlap1_br, overlap2_ul, overlap2_br, flip


class PairHEART(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        # self.MEAN = [0.485]
        # self.STD = [0.229]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')
        self.dataset = PairHEARTDataset(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)
        # dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        super(PairHEART, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,
                                      dist_sampler=dist_sampler)
        # super(PairHEART, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)


class  HEARTDataset(BaseDataSet):
    # **,关键字参数允许你传入0个或任意个含参数名的参数,这些关键字参数在函数内部自动组装为一个dict
    def __init__(self, **kwargs):
        self.num_classes = 4
        self.resizer = transforms.Resize((512, 512))
        # self.datalist = kwargs.pop("datalist")
        #palette 调色
        self.palette = pallete.get_voc_pallete(self.num_classes)
        self.iscut = False
        self.ch = 4
        self.divider = 2
        super(HEARTDataset, self).__init__(**kwargs)
    def _set_files(self):
        return 0

    def __len__(self):
        # 0.1-0.9的实验
        return int(len(os.listdir(self.root))*2//10*self.divider)
    def cut(self,rlabel, image):
        contours, her = cv2.findContours(rlabel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(image.shape)
        bound = cv2.boundingRect(np.array(contours[0]))  # 左上角坐标 x y 矩形的宽，高 w，h
        h, w = image.shape
        x, y, w1, h1 = bound
        # print(x, y, w1, h1)
        x, y, w1, h1 = validcut.boderexpand(x, y, w1, h1,w,h,mode='train')
        # print(x, y, w1, h1)
        image[:y] = 0
        image[y + h1:] = 0
        for i in range(0, h):
            image[i][:x] = 0
            image[i][x + w1:] = 0
        # cv2.drawContours(image, contours, -1, (255), 2)
        image = image.reshape(1,h,w)
        return image
    def _load_data(self, index):
        index = index + 1
        sum = int(len(os.listdir(self.root))//10*self.divider)
        if (index <= sum):
            patient = 'patient' + format(index, '04d')
            image_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ED.mhd'
            label_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ED_gt.mhd'
        else:
            patient = 'patient' + format(index - sum, '04d')
            image_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ES.mhd'
            label_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ES_gt.mhd'
        image_rawfile = sitk.ReadImage(image_path)
        # print(image_path)
        label_rawfile = sitk.ReadImage(label_path)
        image = sitk.GetArrayFromImage(image_rawfile)
        label = sitk.GetArrayFromImage(label_rawfile)
        # image = cv2.resize(image[0], (512, 512))
        # image = image.reshape(1, 512, 512)
        if self.iscut == True:
            image = self.cut(label[0],image[0])
        #切换使用三通道图像的时候使用此代码
        image = cv2.merge((image[0], image[0], image[0]))
        image = torchvision.transforms.ToPILImage()(image)
        label = torchvision.transforms.ToPILImage()(label[0])
        image = self.resizer(image)
        label = self.resizer(label)
        image = self.to_tensor(image)
        label = self.to_tensor(label)
        # image = cv2.resize(image, (512, 512))
        # # print(label.shape)
        # label = cv2.resize(label[0], (512, 512))
        # image = image.reshape(512, 512, 3)
        label = label.reshape(512, 512, 1)
        image = image * 255
        label = label * 255
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.int32)
        image_id = index
        image = image.transpose(1,2,0)
        # plt.subplot(121), plt.imshow(image, cmap="gray")
        # plt.subplot(122), plt.imshow(label, cmap="gray")
        # plt.show()
        # print(111111111111)
        return image, label, image_id



class HEART(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        # self.MEAN = [0.485]
        # self.STD = [0.229]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        # print(sampler_shuffle)
        num_workers = kwargs.pop('num_workers')

        self.dataset = HEARTDataset(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)
        # dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        super(HEART, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,
                                  dist_sampler=dist_sampler)
        # super(HEART, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,)

class  HEARTValidDataset(BaseDataSet):
    # **,关键字参数允许你传入0个或任意个含参数名的参数,这些关键字参数在函数内部自动组装为一个dict
    def __init__(self, **kwargs):
        self.num_classes = 4
        self.resizer = transforms.Resize((512, 512))
        # self.datalist = kwargs.pop("datalist")
        #palette 调色
        self.palette = pallete.get_voc_pallete(self.num_classes)
        self.iscut = False
        # self.border = validcut.readlabel()
        self.ch = 4
        super(HEARTValidDataset, self).__init__(**kwargs)

    def _set_files(self):
        return 0

    def __len__(self):
        return len(os.listdir(self.root))*2
    def cut(self,rlabel, image):
        contours, her = cv2.findContours(rlabel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bound = cv2.boundingRect(np.array(contours[0]))  # 左上角坐标 x y 矩形的宽，高 w，h
        w, h = rlabel.shape
        x, y, w1, h1 = bound
        image[:y] = 0
        image[y + h1:] = 0
        for i in range(0, w):
            image[i][:x] = 0
            image[i][x + w1:] = 0
        # cv2.drawContours(image, contours, -1, (255), 2)
        image = image.reshape(1,w,h)
        return image
    def cutexpand(self,image,bound):
        h, w = image.shape
        x, y, w1, h1 = bound
        x, y, w1, h1 = validcut.boderexpand(x, y, w1, h1,w,h)
        image[:y] = 0
        image[y + h1:] = 0
        for i in range(0, h):
            image[i][:x] = 0
            image[i][x + w1:] = 0
        # cv2.drawContours(image, contours, -1, (255), 2)
        image = image.reshape(1, h, w)
        return image
    def _load_data(self, index):
        index = index + 1
        sum = len(os.listdir(self.root))
        if (index <= sum):
            patient = 'patient' + format(index, '04d')
            # patient = 'patient' + format(index, '04d')
            image_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ED.mhd'
            label_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ED_gt.mhd'
        else:
            patient = 'patient' + format(index - sum, '04d')
            # patient = 'patient' + format(index - sum, '04d')
            image_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ES.mhd'
            label_path = self.root + '/' + patient + '/' + patient + '_'+str(self.ch)+'CH_ES_gt.mhd'
        image_rawfile = sitk.ReadImage(image_path)
        # print(image_path)
        label_rawfile = sitk.ReadImage(label_path)
        image = sitk.GetArrayFromImage(image_rawfile)
        label = sitk.GetArrayFromImage(label_rawfile)
        if index >50:
            bordername = format(index - 50, '04d')+'ES'
        else:
            bordername = format(index, '04d')+'ED'
        image = cv2.resize(image[0],(512,512))
        image = image.reshape(1,512,512)
        if self.iscut == True:
            # image = self.cut(label[0],image[0])
            image = self.cutexpand(image[0],self.border[bordername])
        #切换使用三通道图像的时候使用此代码
        image = cv2.merge((image[0], image[0], image[0]))
        # label = cv2.merge((label[0], label[0], label[0]))
        # print(image.shape)
        image = torchvision.transforms.ToPILImage()(image)
        # print(label.shape)
        label = torchvision.transforms.ToPILImage()(label[0])
        image = self.resizer(image)
        label = self.resizer(label)
        image = self.to_tensor(image)
        label = self.to_tensor(label)
        # image = cv2.resize(image, (512, 512))
        # # print(label.shape)
        # label = cv2.resize(label[0], (512, 512))
        # image = image.reshape(512, 512, 3)
        label = label.reshape(512, 512, 1)
        image = image * 255
        label = label * 255

        # print(image)
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.int32)
        image_id = index
        image = image.transpose(1,2,0)
        # plt.subplot(121),plt.imshow(image,cmap="gray")
        # plt.subplot(122),plt.imshow(label,cmap="gray")
        # plt.show()
        return image, label, image_id



class HEARTValid(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        # self.MEAN = [0.485]
        # self.STD = [0.229]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')

        self.dataset = HEARTValidDataset(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)
        # dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)

        super(HEARTValid, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,
                                  dist_sampler=dist_sampler)
        # super(HEART, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,)


if __name__ == "__main__":
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    batch_size = 8
    ignore_index = 255
    split = "train_supervised"
    data_dir = '/mnt/E05AD4D95AD4AD92/guoziyu/training'
    kwargs = {}
    kwargs['data_dir'] = data_dir
    kwargs['split'] = split
    kwargs['mean'] = MEAN
    kwargs['std'] = STD
    kwargs['ignore_index'] = ignore_index
    heartdataset = HEARTDataset(**kwargs)
    # heartdataset = PairHEARTDataset(**kwargs)
    kwargs['batch_size'] = 8
    kwargs['shuffle'] = True
    kwargs['num_workers'] = 8
    # heart = HEART(kwargs)
    pairheart = PairHEART(kwargs)
    print("数据个数：", len(heartdataset))
