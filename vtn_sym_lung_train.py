def landmark_loss(flow,landmarks1, landmarks2,index):
    spc_s = torch.tensor([0.625, 0.625, 2.5,
    0.645, 0.645, 2.5,
    0.652, 0.652, 2.5,
    0.590, 0.590, 2.5,
    0.647, 0.647, 2.5,
    0.633, 0.633, 2.5,
    0.625, 0.625, 2.5,
    0.586, 0.586, 2.5,
    0.664, 0.664, 2.5,
    0.742, 0.742, 2.5 ]).reshape(10,3).cuda()
    spec = spc_s[index - 1]

    all_dist = []
    flow  = flow[0:1,[2, 1, 0],:,:,:]
    for i in range(landmarks1.shape[1]):
        point = landmarks2[0, i].int()
        move = flow[0, :, point[0] - 1, point[1] - 1, point[2] - 1]
        ori_point = torch.round(point + move)
        dist = landmarks1[0, i] - ori_point
        all_dist.append(dist* spec)

    all_dist = torch.stack(all_dist)
    pt_errs_phys = torch.sqrt(torch.sum(all_dist * all_dist, 1))
    TRE_phys = torch.mean(pt_errs_phys)
    return TRE_phys

def ResizeTransform(flow, target_shape):
    device = flow.device
    _, c, h, w, d = flow.shape
    ratio = torch.FloatTensor([target_shape[0] / h, target_shape[1] / w, target_shape[2] / d])
    ratio = ratio[[2, 1, 0]]
    flow_hr = torch.nn.functional.interpolate(flow, size=(target_shape[0], target_shape[1], target_shape[2]), mode='trilinear',
                                                 align_corners=False) * ratio.to(device=device).view(1, -1, 1, 1, 1)
    return flow_hr


import gpu_utils
import torch
from absl import flags, app
import uflow_net
import numpy as np
import os
from lung_iowa_list import LungDataset
import uflow_utils
import matplotlib.pyplot as plt
import uflow_flags
import torch.backends.cudnn as cudnn
import nibabel as nib


def main(argv):
    cudnn.benchmark = True
    gpu_utils.setup_gpu()

    FLAGS = flags.FLAGS

    num_epochs = FLAGS.num_epochs
    use_lung = FLAGS.dataset == 'lung'

    if use_lung:
        action_channels = 2 if FLAGS.use_minecraft_camera_actions else None
    else:
        action_channels = None

    net = uflow_net.UFlow(num_channels=(1 if use_lung else 1), num_levels=4, use_cost_volume=True,
                          action_channels=action_channels).to(gpu_utils.device)

    # Creat  dataset
    train_dataset = LungDataset(pair = True)
    train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=True,num_samples=50)
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=3,
                                   drop_last=False,
                                   pin_memory=True,
                                   sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(LungDatasetTest(pair = True),
                                   batch_size=1,
                                   num_workers=2,
                                   drop_last=False,
                                   pin_memory=True,
                                   sampler=None)

    optimizer = torch.optim.Adam(list(net._pyramid.parameters()) + list(net._flow_model.parameters()), lr=1e-4)

    os.makedirs('save', exist_ok=True)
    model_save_path = 'save/model.pt'

    loss_history = []
    test_loss_history = []

    if FLAGS.continue_training:
        print('Continuing with model from ' + model_save_path)

        checkpoint = torch.load(model_save_path)

        net.load_state_dict(checkpoint['flow_net'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        loss_history = checkpoint['loss_history']
        test_loss_history = checkpoint['test_loss_history']

    def SaveImage(image, path):
        segmentation = nib.Nifti1Image(image.cpu().detach().numpy(), np.eye(4))
        nib.save(segmentation, path)

    def plot(img, ind=100):
        import matplotlib.pylab as plt
        plt.imshow(img.detach().cpu().numpy()[0, 0, :, :, ind])
        plt.show()
    def generate_warped_image(flow, img):
        size = img.shape[-3:]
        flow_hr = ResizeTransform(flow, size)
        warp = uflow_utils.flow_to_warp(flow_hr)
        warped_img = uflow_utils.resample(img, warp)
        return warped_img
    def get_test_photo_loss():
        with torch.no_grad():
            net.eval()

            losses = []
            for batch in test_loader:
                img1_org = batch['voxel1'].cuda(non_blocking=True).to(torch.float32)
                img2_org = batch['voxel2'].cuda(non_blocking=True).to(torch.float32)
                clamp_min = 80
                clamp_max = 900
                intrange = clamp_max - clamp_min
                img1 = (torch.nn.functional.interpolate(img1_org[:,:,70:470,30:470,:], size=[256, 256, 256], mode='trilinear',
                                                       align_corners=False).clamp_(min=clamp_min,
                                                                                   max=clamp_max)-clamp_min) / intrange
                img2 = (torch.nn.functional.interpolate(img2_org[:,:,70:470,30:470,:], size=[256, 256, 256], mode='trilinear',
                                                       align_corners=False).clamp_(min=clamp_min,
                                                                                   max=clamp_max) -clamp_min)/ intrange
                landmarks1 = batch['landmarks1'].cuda(non_blocking=True).to(torch.float32)
                landmarks2 = batch['landmarks2'].cuda(non_blocking=True).to(torch.float32)
                landmarks2 = landmarks2[..., [1, 0, 2]]
                landmarks1 = landmarks1[..., [1, 0, 2]]
                #ori_shape = batch['voxel_shape']
                flow, pf1, pf2 = net(img1, img2)
                #flow_hr = ResizeTransform(flow[0], img1_org.shape[2:])
                flow_hr = ResizeTransform(flow[0], [400,440,img1_org.shape[-1]])
                index = batch['id2'][0]
                index = int(index.replace('copd', '')[:-1])
                loss = landmark_loss(flow_hr, landmarks2-torch.tensor([70,30,0]).view(1,1,3).cuda(),
                                     landmarks1-torch.tensor([70,30,0]).view(1,1,3).cuda(),index)



                warp = uflow_utils.flow_to_warp(flow_hr)
                warped_f2 = uflow_utils.resample(img2_org[:,:,70:470,30:470,:], warp)

                SaveImage(warped_f2[0, 0], os.path.join('/media/jiaguo/DATA_PUB/RAFT_COPD/network/results',
                                                        batch['id1'][0]+'_'+batch['id2'][0]))
                SaveImage(img1_org[0,0,70:470,30:470,:], os.path.join('/media/jiaguo/DATA_PUB/RAFT_COPD/network/results',batch['id1'][0]))
                SaveImage(img2_org[0,0,70:470,30:470,:], os.path.join('/media/jiaguo/DATA_PUB/RAFT_COPD/network/results',batch['id2'][0]))

                warped_image_level0 = generate_warped_image(flow[-1], img2_org[:,:,70:470,30:470,:])
                SaveImage(warped_image_level0[0, 0], os.path.join('/media/jiaguo/DATA_PUB/RAFT_COPD/network/results',
                                                        'level0'+batch['id1'][0]+'_'+batch['id2'][0]))
                warped_image_level1 = generate_warped_image(flow[-2], img2_org[:,:,70:470,30:470,:])
                SaveImage(warped_image_level1[0, 0], os.path.join('/media/jiaguo/DATA_PUB/RAFT_COPD/network/results',
                                                        'level1'+batch['id1'][0]+'_'+batch['id2'][0]))
                warped_image_level2 = generate_warped_image(flow[-3], img2_org[:,:,70:470,30:470,:])
                SaveImage(warped_image_level2[0, 0], os.path.join('/media/jiaguo/DATA_PUB/RAFT_COPD/network/results',
                                                        'level2'+batch['id1'][0]+'_'+batch['id2'][0]))
                warped_image_level3 = generate_warped_image(flow[-4], img2_org[:,:,70:470,30:470,:])
                SaveImage(warped_image_level3[0, 0], os.path.join('/media/jiaguo/DATA_PUB/RAFT_COPD/network/results',
                                                        'level3'+batch['id1'][0]+'_'+batch['id2'][0]))

                losses.append(loss.item())
                print(batch['id2'][0] + 'loss=%.5f' % (loss))

            loss = np.mean(losses)
            print('mean loss=%.5f' % (loss))
            #show_results(net, test_loader, epoch, 2)
            return loss

    test_loss = test_loss_history[-1] if test_loss_history else float("nan")
    #get_test_photo_loss()
    ncc = NCC_fft()
    for epoch in range(num_epochs):
        net.train()
        losses = []
        for batch in train_loader:
            img1 = batch['voxel1_hr'].cuda(non_blocking=True).to(torch.float32)
            img2 = batch['voxel2_hr'].cuda(non_blocking=True).to(torch.float32)

            clamp_min = 80
            clamp_max = 900
            intrange = clamp_max - clamp_min
            img1 = (torch.nn.functional.interpolate(img1[:,:,70:470,30:470,:], size=[256, 256, 256], mode='trilinear',
                                                   align_corners=False).clamp_(min=clamp_min,
                                                                               max=clamp_max)-clamp_min) / intrange
            img2 = (torch.nn.functional.interpolate(img2[:,:,70:470,30:470,:], size=[256, 256, 256], mode='trilinear',
                                                   align_corners=False).clamp_(min=clamp_min,
                                                                               max=clamp_max)-clamp_min) / intrange

            flow, pf1, pf2 = net(img1, img2)
            loss = net.compute_loss(img1, img2, pf1, pf2, flow,ncc)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(list(net._pyramid.parameters()) + list(net._flow_model.parameters()),2)
            optimizer.step()

        loss = np.mean(losses)
        loss_history.append(loss)
        test_loss = get_test_photo_loss()
        test_loss_history.append(test_loss)
        '''print(
            "Epoch {} loss : {:.2f}e-6, pure test photo loss: {:.2f}e-6".format(epoch, loss * 1e6, test_loss * 1e6))
        '''
        print(
            "Epoch {}".format(epoch))
        if epoch % 2 == 1:
            print("Saving model to " + model_save_path)

            torch.save({
                'flow_net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': loss_history,
                'test_loss_history': test_loss_history,
            }, model_save_path)

        plt.figure()
        plt.plot(range(len(loss_history)), [x * 1e6 for x in loss_history])
        plt.plot(range(len(test_loss_history)), [x * 1e6 for x in test_loss_history])
        plt.legend(['training loss', 'test photometric loss'])
        plt.show()

class LungDatasetTest(torch.utils.data.Dataset):
    def __init__(self, img_path='/media/jiaguo/DATA_PUB/REG_COPD/data/', landmarks_path ='/media/jiaguo/DATA_PUB/REG_COPD/landmarks', pair=True, is_training = False):
        super(LungDatasetTest, self).__init__()
        self.img_path = img_path
        self._pair = pair
        file_names = os.listdir(self.img_path)
        file_names.sort()
        self._file_names = []
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
        names = self._file_names[index]
        img_path = [os.path.join(self.img_path, i) for i in names]

        item_name = [i.split(".")[0] for i in names]

        img1, img2 = self._fetch_data(img_path)
        img1 = img1[np.newaxis,...]
        img2 = img2[np.newaxis,...]


        output_dict = dict( voxel1 = img1, voxel2 = img2, id1=str(item_name[0]),id2=str(item_name[1]))

        # load landmarks
        subject_name = item_name[0][:-1]+'_'
        if 'e' in  item_name[0]:
            phase_name = '_eBH_'
            phase_name2 = '_iBH_'
        else:
            phase_name = '_iBH_'
            phase_name2 = '_eBH_'

        landmarks_name  = [ i for i in self.landmarks_names if  subject_name in i and phase_name in i][0]
        landmarks_name2  = [ i for i in self.landmarks_names if  subject_name in i and phase_name2 in i][0]
        landmarks2= []
        with open(os.path.join( self.landmarks_path,landmarks_name2),'r') as f:
            for line in f:
                processed_line = line.split('\t')[:-1]
                processed_line = [float(i) for i in processed_line]
                landmarks2.append(processed_line)
        landmarks = []
        with open(os.path.join(self.landmarks_path, landmarks_name), 'r') as f:
            for line in f:
                processed_line = line.replace('\n','').split('\t')
                processed_line = [float(i) for i in processed_line]
                landmarks.append(processed_line)
        landmarks2 = np.array(landmarks2)
        landmarks = np.array(landmarks)
        output_dict.update(landmarks1=landmarks,landmarks2=landmarks2)
        return output_dict
    def _fetch_data(self, img_path, Format = 'nii'):
        if Format=='mat':
            img1 = self._open_image(img_path[0])
            img2 = self._open_image(img_path[1])
        else:
            img1 = nib.load(img_path[0]).get_fdata()

            img2 = nib.load(img_path[1]).get_fdata()

        return img1, img2

if __name__ == '__main__':
    a = torch.arange(256*256*256).view(1,1,256,256,256)
    from loss_metrics import NCC_fft
    #ncc =NCC_fft()
    #b = ncc(a,a)
    app.run(main)

