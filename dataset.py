from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import cv2
import torch
import numpy as np
import random
import scipy.io as scio
import cv2
import h5py

class Dataset_syn(Dataset):
    def __init__(self, file_path, data_key, patch_size=64, ratio=4, crop_ratio=0.8, mc=3, iters=1000):
        super(Dataset_syn, self).__init__()
        self.file_path = file_path
        self.patch_size = patch_size
        self.ratio = ratio
        self.crop_ratio = crop_ratio
        self.mc = 3
        self.iters = iters

        if data_key == 'chikusei':
            self.data_mat = np.array(h5py.File(file_path,'r')[data_key]).transpose(1,2,0)
        else:
            self.data_mat = scio.loadmat(file_path)[data_key]
        self.size = self.data_mat.shape
        self.norm_data = self.norm(self.data_mat)
        self.gap_bands = self.data_mat.shape[-1] / (mc-1.0)


        self.hrmsi = self.norm_data[:,:,0][:,:,np.newaxis]
        for i in range(1, mc-1):
            self.hrmsi = np.concatenate((self.hrmsi, self.norm_data[:,:,int(self.gap_bands*i)][:,:,np.newaxis]), axis=2)
        self.hrmsi = np.concatenate((self.hrmsi, self.norm_data[:,:,mc-1][:,:,np.newaxis],), axis=2)

        self.gt = self.norm_data.copy()

        self.train_ref,self.train_hrmsi,self.test_ref,self.test_lrhsi,self.test_hrmsi = self.crop(crop_ratio)

        print(self.train_ref.shape,self.train_hrmsi.shape,self.test_ref.shape,self.test_lrhsi.shape,self.test_hrmsi.shape)

    def norm(self,array):
        return (array-np.min(array))/(np.max(array)-np.min(array))

    def blur(self, ref):
        h,w,c = ref.shape
        blur = cv2.GaussianBlur(ref.copy(), (5,5), 2)
        lrhsi = cv2.resize(blur, (h//self.ratio, w//self.ratio))
        return lrhsi

    def crop(self, crop_ratio=0.8):
        patch_size = self.patch_size
        h,w,c = self.gt.shape
        clip_rows = h//patch_size
        clip_cols = w//patch_size
        train_rows = int(crop_ratio*clip_rows)
        train_cols = int(crop_ratio*clip_cols)
        test_ref,test_lrhsi,test_hrmsi = [], [], []

        train_h = train_rows*patch_size
        train_w = train_cols*patch_size
        train_ref = self.gt[:train_h, :train_w,:]
        train_hrmsi = self.hrmsi[:train_h, :train_w,:]


        for i in range(clip_rows):
            for j in range(clip_cols):
                start_y = i*patch_size
                start_x = j*patch_size
                ref_clip = self.gt[start_y:start_y+patch_size,
                                   start_x:start_x+patch_size, :]

                hsi_clip = self.blur(ref_clip)
                msi_clip = self.hrmsi[start_y:start_y+patch_size,
                                   start_x:start_x+patch_size, :]
                if (i >= train_rows) or (j >=train_cols):
                    test_ref.append(ref_clip[None])
                    test_lrhsi.append(hsi_clip[None])
                    test_hrmsi.append(msi_clip[None])

        test_ref = np.concatenate(test_ref, 0)
        test_lrhsi = np.concatenate(test_lrhsi, 0)
        test_hrmsi = np.concatenate(test_hrmsi, 0)

        return train_ref,train_hrmsi,test_ref,test_lrhsi,test_hrmsi

    def get_random_train(self, batch_size=1):
        h,w,c = self.gt.shape
        size = self.patch_size
        gt_b = []
        lrhsi_b = []
        hrmsi_b = []
        for i in range(batch_size):
            h_str = random.randint(0, h-size-1)
            w_str = random.randint(0, w-size-1)
            
            gt = self.gt[h_str:h_str+self.patch_size, w_str:w_str+self.patch_size, :]
            lrhsi = self.blur(gt)
            hrmsi = self.hrmsi[h_str:h_str+self.patch_size, w_str:w_str+self.patch_size, :]

            gt = torch.from_numpy(gt).permute(2,0,1)
            lrhsi = torch.from_numpy(lrhsi).permute(2,0,1)
            hrmsi = torch.from_numpy(hrmsi).permute(2,0,1)

            gt_b.append(gt.unsqueeze(0))
            lrhsi_b.append(lrhsi.unsqueeze(0))
            hrmsi_b.append(hrmsi.unsqueeze(0))
        
        gt_b = torch.concat(gt_b, 0)
        lrhsi_b = torch.concat(lrhsi_b, 0)
        hrmsi_b = torch.concat(hrmsi_b, 0)

        return gt_b, lrhsi_b, hrmsi_b

    def get_test(self):
        gt = torch.from_numpy(self.test_ref).permute(0,3,1,2)
        lrhsi = torch.from_numpy(self.test_lrhsi).permute(0,3,1,2)
        hrmsi = torch.from_numpy(self.test_hrmsi).permute(0,3,1,2)
        return gt, lrhsi, hrmsi

    def get_test_numpy(self):
        gt = self.test_ref
        lrhsi = self.test_lrhsi
        hrmsi = self.test_hrmsi
        return gt, lrhsi, hrmsi

    def __getitem__(self):
        return None

    def __len__(self):
        if self.iters is not None:
            return self.iters

import os
import tifffile  
class Dataset_real(Dataset):
    def __init__(self, file_path, patch_size=64, iters=1000):
        super(Dataset_real, self).__init__()
        self.file_path = file_path
        self.patch_size = patch_size
        self.iters = iters

        self.train_ref = self.read_tif(os.path.join(file_path,'sr_deep_model_data','EeteS_EnMAP_10m_deep_train.tif'))
        self.train_lrhsi = self.read_tif(os.path.join(file_path,'sr_deep_model_data','EeteS_EnMAP_30m_deep_train.tif'))
        self.train_hrmsi = self.read_tif(os.path.join(file_path,'sr_deep_model_data','EeteS_Sentinel_2_10m_deep_train.tif'))
        self.mc = self.train_hrmsi.shape[-1]
        self.ratio = self.train_hrmsi.shape[0]//self.train_lrhsi.shape[0]
        print(self.train_ref.shape, self.train_lrhsi.shape, self.train_hrmsi.shape)

        self.valid_ref = self.read_tif(os.path.join(file_path,'sr_deep_model_data','EeteS_EnMAP_10m_deep_valid.tif'))
        self.valid_lrhsi = self.read_tif(os.path.join(file_path,'sr_deep_model_data','EeteS_EnMAP_30m_deep_valid.tif'))
        self.valid_hrmsi = self.read_tif(os.path.join(file_path,'sr_deep_model_data','EeteS_Sentinel_2_10m_deep_valid.tif'))

        max_ = np.max([self.train_ref.max(), self.train_lrhsi.max(),
                       self.train_hrmsi.max(), self.valid_ref.max(),
                       self.valid_lrhsi.max(), self.train_hrmsi.max()])

        min_ = np.min([self.train_ref.min(), self.train_lrhsi.min(),
                       self.train_hrmsi.min(), self.valid_ref.min(),
                       self.valid_lrhsi.min(), self.train_hrmsi.min()])

        self.train_ref = self.norm(self.train_ref, max_, min_)
        self.train_lrhsi = self.norm(self.train_lrhsi, max_, min_)
        self.train_hrmsi = self.norm(self.train_hrmsi, max_, min_)

        self.valid_ref = self.norm(self.valid_ref, max_, min_)
        self.valid_lrhsi = self.norm(self.valid_lrhsi, max_, min_)
        self.valid_hrmsi = self.norm(self.valid_hrmsi, max_, min_)

    def norm(self, data, max_, min_):
        return (data-min_)/(max_-min_)

    def read_tif(self, tif_data_path):
        img = tifffile.imread(tif_data_path)
        return img 

    def get_random_train(self, batch_size=1):
        lrh,lrw,_ = self.train_lrhsi.shape
        lr_ps = self.patch_size//self.ratio
        ps = self.patch_size
        gt_b = []
        lrhsi_b = []
        hrmsi_b = []
        for i in range(batch_size):
            lrh_str = random.randint(0, lrh-lr_ps-1)
            lrw_str = random.randint(0, lrw-lr_ps-1)
            # print(self.train_lrhsi.shape, lrh_str, lrw_str, lr_ps)
            lrhsi = self.train_lrhsi[lrh_str:lrh_str+lr_ps, lrw_str:lrw_str+lr_ps, :]

            hrh_str = lrh_str*self.ratio
            hrw_str = lrw_str*self.ratio

            # print(self.train_hrmsi.shape, hrh_str, hrw_str, ps)
            gt = self.train_ref[hrh_str:hrh_str+ps, hrw_str:hrw_str+ps, :]
            hrmsi = self.train_hrmsi[hrh_str:hrh_str+ps, hrw_str:hrw_str+ps, :]

            gt = torch.from_numpy(gt).permute(2,0,1)
            lrhsi = torch.from_numpy(lrhsi).permute(2,0,1)
            hrmsi = torch.from_numpy(hrmsi).permute(2,0,1)

            gt_b.append(gt.unsqueeze(0))
            lrhsi_b.append(lrhsi.unsqueeze(0))
            hrmsi_b.append(hrmsi.unsqueeze(0))

        gt_b = torch.concat(gt_b, 0)
        lrhsi_b = torch.concat(lrhsi_b, 0)
        hrmsi_b = torch.concat(hrmsi_b, 0)

        return gt_b, lrhsi_b, hrmsi_b

    def get_test(self, crop=48):
        gt = torch.from_numpy(self.valid_ref[None]).permute(0,3,1,2)
        lrhsi = torch.from_numpy(self.valid_lrhsi[None]).permute(0,3,1,2)
        hrmsi = torch.from_numpy(self.valid_hrmsi[None]).permute(0,3,1,2)
        if crop is not None:
            _,_,h,w = gt.shape
            clip_rows = (h//crop)
            clip_cols = (w//crop)
            gt = gt[:,:,:clip_rows*crop,:clip_cols*crop]
            lrhsi = lrhsi[:,:,:clip_rows*int(crop/3),:clip_cols*int(crop/3)]
            hrmsi = hrmsi[:,:,:clip_rows*crop,:clip_cols*crop]
        return gt, lrhsi, hrmsi

    def get_test_numpy(self):
        gt = self.valid_ref
        lrhsi = self.valid_lrhsi
        hrmsi = self.valid_hrmsi
        return gt, lrhsi, hrmsi

    def __getitem__(self):
        return None

    def __len__(self):
        if self.iters is not None:
            return self.iters


if __name__ == '__main__':
    file_path = './data/PaviaC/Pavia.mat'
    data_key = 'pavia'
    dataset = Dataset_syn(file_path, data_key, patch_size=64, ratio=4, crop_ratio=0.8, mc=3, iters=1000)
