import gpu_utils
import torch
from absl import flags, app
import rnn_net
import numpy as np
import os
from lung_iowa_list import LungDataset
import rnn_utils
import matplotlib.pyplot as plt
import rnn_flags
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

if __name__ == '__main__':
    app.run(main)

