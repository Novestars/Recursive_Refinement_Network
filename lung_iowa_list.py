import os
import numpy as np
import scipy
import scipy.ndimage
import scipy.io
import nibabel as nib
import torch.utils.data as data
import torch
class LungDataset(data.Dataset):
    def __init__(self, hr_img_path='/DATA/DATA_HXZ/RAFT_COPD/data/',
                 mask_path =  '/DATA/DATA_HXZ/vessel/'
                 , landmarks_path='/DATA/DATA_HXZ/RAFT_COPD/landmarks/'
                 , patch = True
                 , pair=False, image_size = 400, is_training = True):
        super(LungDataset, self).__init__()
        self.is_training = is_training
        self._hr_img_path = hr_img_path
        self._pair = pair
        file_names = self._get_file_names()
        file_names.sort()
        self._file_names = []
        self.image_size = image_size
        self.patch = patch
        self._mask_path = mask_path
        ordered = True


        if not pair:
            for i, d1 in enumerate(file_names):
                for j, d2 in enumerate(file_names):
                    if i != j:
                        if ordered or i < j:
                            self._file_names.append((d1, d2))
        else:
            d1 = None
            for d2 in file_names:
                if d1 is None:
                    d1 = d2
                else:
                    self._file_names.append((d1, d2))
                    self._file_names.append((d2, d1))
                    d1 = None
        self._file_names = [self._file_names[i] for i in range(0, len(self._file_names)) if i % 2 == 1]
        self._file_length = len(self._file_names)
        self.landmarks_names = os.listdir(landmarks_path)
        self.landmarks_names.sort()
        self.landmarks_path =landmarks_path


    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        valid_mask = np.ones([6], dtype=np.bool)

        names = self._file_names[index]
        hr_img_path = [os.path.join(self._hr_img_path, i) for i in names]

        item_name = [i.split(".")[0] for i in names]
        '''if self.patch:
            first_stage_name = item_name[1]+'_'+item_name[0]+'.nii'
            img_path[1] = os.path.join(self.first_stage_output, first_stage_name)'''

        img1, img2 = self._fetch_data(hr_img_path)
        voxel_shape = img1.shape
        clamp_min = 80
        clamp_max = 900
        int_range = clamp_max - clamp_min

        '''img1 = torch.nn.functional.interpolate(torch.tensor(img1[70:470,30:470,:]).unsqueeze(0).unsqueeze(0), size=[256, 256, 256], mode='trilinear',
                                               align_corners=False).clamp_(
            min=clamp_min, max=clamp_max) / int_range
        img2 = torch.nn.functional.interpolate(torch.tensor(img2[70:470,30:470,:]).unsqueeze(0).unsqueeze(0), size=[256, 256, 256], mode='trilinear',
                                               align_corners=False).clamp_(
            min=clamp_min, max=clamp_max) / int_range

        img1 = img1[0,...]
        img2 = img2[0,...]'''

        img1 = img1[np.newaxis,...]
        img2 = img2[np.newaxis,...]


        output_dict = dict( voxel1_hr = img1, voxel2_hr = img2, id1=str(item_name[0]),id2=str(item_name[1]))


        output_dict['point1'] = np.ones(
            (1, np.sum(valid_mask), 3), dtype=np.float32) * (-1)
        output_dict['point2'] = np.ones(
            (1, np.sum(valid_mask), 3), dtype=np.float32) * (-1)
        # load landmarks
        '''subject_name = item_name[0][:-1]+'_'
        if 'e' in  item_name[0]:
            phase_name = '_eBH_'
            phase_name2 = '_iBH_'
        else:
            phase_name = '_iBH_'
            phase_name2 = '_eBH_'

        landmarks_name  = [ i for i in self.landmarks_names if  subject_name in i and phase_name in i][0]
        landmarks_name2  = [ i for i in self.landmarks_names if  subject_name in i and phase_name2 in i][0]
        landmarks= []
        with open(os.path.join( self.landmarks_path,landmarks_name),'r') as f:
            for line in f:
                processed_line = line.split('\t')[:-1]
                processed_line = [float(i) for i in processed_line]
                landmarks.append(processed_line)
        landmarks2 = []
        with open(os.path.join(self.landmarks_path, landmarks_name2), 'r') as f:
            for line in f:
                processed_line = line.replace('\n','').split('\t')
                processed_line = [float(i) for i in processed_line]
                landmarks2.append(processed_line)
        landmarks2 = np.array(landmarks2)
        landmarks = np.array(landmarks)
        output_dict.update(landmarks1=landmarks,landmarks2=landmarks2)'''
        output_dict.update(voxel_shape = voxel_shape)
        return output_dict

    def _fetch_data(self, img_path, Format = 'nii'):
        if Format=='mat':
            img1 = self._open_image(img_path[0])
            img2 = self._open_image(img_path[1])
        else:
            img1 = self._open_image_nii(img_path[0])
            img2 = self._open_image_nii(img_path[1])

        return img1, img2

    def _get_file_names(self):

        file_names = os.listdir(self._hr_img_path)

        return file_names


    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath):
        # cv2: B G R
        # h w c
        img =  scipy.io.loadmat(filepath)['img']

        return img
    @staticmethod
    def _open_image_nii(filepath):
        # cv2: B G R
        # h w c
        img = nib.load(filepath).get_fdata()

        return img



if __name__ == "__main__":
    bd = LungDataset(pair = True)
    for i in range(405):
        bd[i]             
