# import mmcv
from doctest import FAIL_FAST
from operator import is_
from pickle import FALSE
import decord
from PIL import Image, ImageEnhance
from pip import main
import torch
import numpy as np
import torchvision.transforms as transforms
import json
import os

identity = lambda x: x
transformtypedict = dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class SubVdieoDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, random_select=False,
                 num_segments=None, with_flow=False):
        self.sub_meta = sub_meta
        # self.video_list = [x.strip().split(' ') for x in open(sub_meta)]
        self.with_flow = with_flow
        if with_flow == True:
            self.image_tmpl1 = 'x_{:06d}.jpg'
            self.image_tmpl2 = 'y_{:06d}.jpg'
        if True:
            # self.image_tmpl = 'img_{:05d}.jpg'
            self.image_tmpl = '{:06d}.jpg'
        else:
            # self.image_tmpl = 'img_{:05d}.png'
            self.image_tmpl = '{:06d}.jpg'
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.random_select = random_select
        self.num_segments = num_segments

    def __getitem__(self, i):
        # image_path = os.path.join( self.sub_meta[i])
        assert len(self.sub_meta[i]) == 2
        full_path = self.sub_meta[i][0]
        num_frames = self.sub_meta[i][1]
        num_segments = self.num_segments
        if self.with_flow == True:
            length = 5  # 5 frames
            num_segments = num_segments // 3
            average_duration = (num_frames - length + 1) // num_segments
            if average_duration > 0:
                frame_id = np.multiply(list(range(num_segments)), average_duration)
                frame_id = frame_id + np.random.randint(average_duration, size=num_segments)
            elif num_frames > num_segments:
                frame_id = np.sort(np.random.randint(num_frames - length + 1, size=num_segments))
            else:
                frame_id = np.zeros((num_segments,))

            if frame_id[0] == 0:
                frame_id[0] = frame_id[0] + 1  # idx >= 1

            img_group = []
            for k in range(num_segments):
                p = int(k)
                channel_ar = []
                for i in range(length):
                    x_img_path = os.path.join(full_path, self.image_tmpl1.format(frame_id[p] + i))
                    x_img = Image.open(x_img_path)
                    y_img_path = os.path.join(full_path, self.image_tmpl2.format(frame_id[p] + i))
                    y_img = Image.open(y_img_path)
                    # print(type(y_img))
                    x_img = self.transform(x_img)
                    y_img = self.transform(y_img)
                    xy_img = torch.cat([x_img, y_img], dim=0)  # 2 224 224
                    # seg_imgs = [x_img, y_img]
                    channel_ar.append(xy_img)
                channel_ar = torch.cat(channel_ar, dim=0)
                img_group.append(channel_ar)

        else:
            if self.random_select and num_frames > 8:  # random sample
                # frame_id = np.random.randint(num_frames)
                average_duration = num_frames // num_segments
                frame_id = np.multiply(list(range(num_segments)), average_duration)
                frame_id = frame_id + np.random.randint(average_duration, size=num_segments)
            else:
                # frame_id = num_frames//2
                tick = num_frames / float(num_segments)
                frame_id = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
            frame_id = frame_id + 1  # idx >= 1

            img_group = []
            for k in range(num_segments):
                img_path = os.path.join(full_path, self.image_tmpl.format(frame_id[k]))
                img = Image.open(img_path)
                img = self.transform(img)  # RGB: 3x244x244
                img_group.append(img)

        img_group = torch.stack(img_group, 0)
        target = self.target_transform(self.cl)
        # print('ok',image_path)
        return img_group, target  # , full_path

    def __len__(self):
        return len(self.sub_meta)


class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


class SetDataManager:
    def __init__(self, image_size, n_way, n_support, n_query, num_segments, n_eposide=100, with_flow=False):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.with_flow = with_flow

        self.trans_loader = TransformLoader(image_size, is_flow=with_flow)
        self.num_segments = num_segments

    def get_data_loader(self, data_file, aug, with_flow):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug, is_flow=with_flow)
        dataset = SetDataset(data_file, self.batch_size, transform, random_select=aug, num_segments=self.num_segments,
                             with_flow=with_flow)  # video
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(batch_sampler=sampler, num_workers=8, pin_memory=False)  ########    pin_memory = True
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class IdentityTransform(object):

    def __call__(self, data):
        return data


class TransformLoader:
    def __init__(self, image_size,
                 # normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 normalize_param=dict(mean=[0.376, 0.401, 0.431], std=[0.224, 0.229, 0.235]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4),
                 is_flow=False):
        self.image_size = image_size
        self.is_flow = is_flow
        if self.is_flow:
            self.normalize_param = dict(mean=[0.5], std=[0.5])
        else:
            self.normalize_param = normalize_param

        self.jitter_param = jitter_param

    def parse_transform(self, transform_type, is_flow):
        if transform_type == 'ImageJitter':
            method = ImageJitter(self.jitter_param, is_flow)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False, is_flow=False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            if is_flow:
                transform_list = ['ToTensor']
            else:
                transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']


        transform_funcs = [self.parse_transform(x, is_flow) for x in transform_list]
        transform = transforms.Compose(transform_funcs)

        return transform


class SetDataset:  # frames
    def __init__(self, data_file, batch_size, transform, random_select=False, num_segments=None, with_flow=False):

        self.video_list = [x.strip().split(' ') for x in open(data_file)]


        self.cl_list = np.zeros(len(self.video_list), dtype=int)
        for i in range(len(self.video_list)):
            self.cl_list[i] = self.video_list[i][2]
        self.cl_list = np.unique(self.cl_list).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x in range(len(self.video_list)):
            root_path = self.video_list[x][0]
            if with_flow == True:
                num_frames = int(int(self.video_list[x][1]) / 2)  
            else:
                num_frames = int(self.video_list[x][1])
            label = int(self.video_list[x][2])
            self.sub_meta[label].append([root_path, num_frames])

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubVdieoDataset(self.sub_meta[cl], cl, transform=transform, random_select=random_select,
                                          num_segments=num_segments, with_flow=with_flow)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class ImageJitter(object):
    def __init__(self, transformdict, is_flow=False):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]
        self.is_flow = is_flow

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            if not self.is_flow:
                out = transformer(out).enhance(r).convert('RGB')
            else:
                out = transformer(out).enhance(r)

        return out


class SubVideo_RGB_FLOW_Dataset:
    def __init__(self, sub_meta_rgb, sub_meta_flow, cl, transform=transforms.ToTensor(),
                 transform_flow=transforms.ToTensor(), target_transform=identity, random_select=False,
                 num_segments=None):
        self.sub_meta_rgb = sub_meta_rgb
        self.sub_meta_flow = sub_meta_flow

        self.image_tmplx = 'x_{:06d}.jpg'
        self.image_tmply = 'y_{:06d}.jpg'
        self.image_tmpl_rgb = '{:06d}.jpg'

        self.cl = cl
        self.transform = transform
        self.transform_flow = transform_flow
        self.target_transform = target_transform
        self.random_select = random_select
        self.num_segments = num_segments

    def __getitem__(self, i):

        assert len(self.sub_meta_rgb[i]) == 2

        img_group_RGB = self.readRGB(i)

        img_group_RGB = torch.stack(img_group_RGB, 0)

        target = self.target_transform(self.cl)

        return img_group_RGB, target  


    def readFlow(self, i):
        full_path = self.sub_meta_flow[i][0]
        num_frames = self.sub_meta_flow[i][1]
        num_segments = self.num_segments

        length = 2  # 5 frames
        num_segments = num_segments
        average_duration = (num_frames - length + 1) // num_segments
        if average_duration > 0:
            frame_id = np.multiply(list(range(num_segments)), average_duration)
            frame_id = frame_id + np.random.randint(average_duration, size=num_segments)
        elif num_frames > num_segments:
            frame_id = np.sort(np.random.randint(num_frames - length + 1, size=num_segments))
        else:
            frame_id = np.zeros((num_segments,))

        if frame_id[0] == 0:
            frame_id[0] = frame_id[0] + 1  # idx >= 1

        img_group = []
        for k in range(num_segments):
            p = int(k)
            channel_ar = []
            for i in range(length):
                x_img_path = os.path.join(full_path, self.image_tmplx.format(frame_id[p] + i))
                x_img = Image.open(x_img_path)
                y_img_path = os.path.join(full_path, self.image_tmply.format(frame_id[p] + i))
                y_img = Image.open(y_img_path)
                x_img = self.transform_flow(x_img)
                y_img = self.transform_flow(y_img)
                xy_img = torch.cat([x_img, y_img], dim=0)  
                channel_ar.append(xy_img)
            channel_ar = torch.cat(channel_ar, dim=0)
            img_group.append(channel_ar)
        return img_group

    def readRGB(self, i):
        full_path = self.sub_meta_rgb[i][0]
        num_frames = self.sub_meta_rgb[i][1]
        num_segments = self.num_segments
        full_path_flow = self.sub_meta_flow[i][0]
       
        average_duration = num_frames // num_segments
        frame_id = np.multiply(list(range(num_segments)), average_duration)
        frame_id = frame_id + np.random.randint(average_duration, size=num_segments)
        frame_id = frame_id + 1  # idx >= 1


        rgb_mean = [0.376, 0.401, 0.431]
        rgb_std = [0.224, 0.229, 0.235]

        gray_mean = [sum(rgb_mean) / 3.0]
        gray_std = [sum(rgb_std) / 3.0]
        transform_gray = transforms.Compose([
                    transforms.RandomResizedCrop(size=(224,224)), 
                    transforms.RandomHorizontalFlip(), 
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=gray_mean, std=gray_std)
                    ])

        img_group = []

        for k in range(self.num_segments):
            img_path = os.path.join(full_path, self.image_tmpl_rgb.format(frame_id[k]))
            img = Image.open(img_path)

            img_gray = transform_gray(img)

            img = self.transform(img)
            if frame_id[k] >= num_frames:
                frame_id[k] = num_frames - 1
            if frame_id[k] == 1:
                frame_id[k] = frame_id[k] + 1

            #flow
           
            x1_img_path = os.path.join(full_path_flow, self.image_tmplx.format(frame_id[k] - 1))
            y1_img_path = os.path.join(full_path_flow, self.image_tmply.format(frame_id[k] - 1))
            x2_img_path = os.path.join(full_path_flow, self.image_tmplx.format(frame_id[k]))
            y2_img_path = os.path.join(full_path_flow, self.image_tmply.format(frame_id[k]))
            x1_img = Image.open(x1_img_path)
            y1_img = Image.open(y1_img_path)
            x2_img = Image.open(x2_img_path)
            y2_img = Image.open(y2_img_path)
            x1_img = self.transform_flow(x1_img)
            y1_img = self.transform_flow(y1_img)
            x2_img = self.transform_flow(x2_img)
            y2_img = self.transform_flow(y2_img)
            xy_img = torch.cat([x1_img, y1_img, x2_img, y2_img], dim=0)

             #gray
            img_path = os.path.join(full_path, self.image_tmpl_rgb.format(frame_id[k] - 1))
            img1 = Image.open(img_path)
            img1 = transform_gray(img1)


            img_path = os.path.join(full_path, self.image_tmpl_rgb.format(frame_id[k] + 1))
            img2 = Image.open(img_path)
            img2 = transform_gray(img2)

            xyz_img = torch.cat([img, img1, img_gray, img2, xy_img], dim=0)
            img_group.append(xyz_img)

        return img_group

    def __len__(self):
        return len(self.sub_meta_rgb)


class SetDataset_RGB_FLOW:  # frames
    def __init__(self, data_file, data_file_flow, batch_size, transform, transform_flow, random_select=False,
                 num_segments=None):

        self.video_list = [x.strip().split(' ') for x in open(data_file)]
        self.video_list_flow = [x.strip().split(' ') for x in open(data_file_flow)]

        self.cl_list = np.zeros(len(self.video_list), dtype=int)
        for i in range(len(self.video_list)):
            self.cl_list[i] = self.video_list[i][2]
        self.cl_list = np.unique(self.cl_list).tolist()
        self.cl_list_flow = np.zeros(len(self.video_list_flow), dtype=int)
        for i in range(len(self.video_list_flow)):
            self.cl_list_flow[i] = self.video_list_flow[i][2]
        self.cl_list_flow = np.unique(self.cl_list_flow).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []
        self.sub_meta_flow = {}
        for cl in self.cl_list_flow:
            self.sub_meta_flow[cl] = []

        # for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
        # self.sub_meta[y].append(x)
        for x in range(len(self.video_list)):
            root_path = self.video_list[x][0]
            num_frames = int(self.video_list[x][1])
            label = int(self.video_list[x][2])
            self.sub_meta[label].append([root_path, num_frames])

        for x in range(len(self.video_list_flow)):
            root_path = self.video_list_flow[x][0]
            num_frames = int(int(self.video_list_flow[x][1]) / 2)  # flow_nums分x和y
            label = int(self.video_list_flow[x][2])
            self.sub_meta_flow[label].append([root_path, num_frames])

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubVideo_RGB_FLOW_Dataset(self.sub_meta[cl], self.sub_meta_flow[cl], cl, transform=transform,
                                                    transform_flow=transform_flow, random_select=random_select,
                                                    num_segments=num_segments)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SetDataManager_RGB_FLOW:
    def __init__(self, image_size, n_way, n_support, n_query, num_segments, n_eposide=100):
        super(SetDataManager_RGB_FLOW, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.num_segments = num_segments
        self.trans_loader_rgb = TransformLoader(image_size, is_flow=False)
        self.trans_loader_flow = TransformLoader(image_size, is_flow=True)

    def get_data_loader(self, data_file, data_file_flow, aug):  # parameters that would change on train/val set
        transform = self.trans_loader_rgb.get_composed_transform(aug, is_flow=False)
        transform_flow = self.trans_loader_flow.get_composed_transform(aug = False, is_flow=True)
        dataset = SetDataset_RGB_FLOW(data_file, data_file_flow, self.batch_size, transform, transform_flow,
                                      random_select=aug, num_segments=self.num_segments)  # video
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(batch_sampler=sampler, num_workers=8, pin_memory=False)  ########    pin_memory = True
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


def test_dataloader():
    para = ''
    dataset = SubVdieoDataset()

    pass

