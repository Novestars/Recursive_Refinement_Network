import uflow_model
import uflow_utils
import torch.nn as nn

class UFlow(nn.Module):
    def __init__(self, num_channels=3, num_levels = 5, use_cost_volume=True, action_channels = 0):
        super(UFlow, self).__init__()

        self._pyramid = uflow_model.PWCFeaturePyramid(num_levels=num_levels, num_channels=num_channels)
        self._flow_model = uflow_model.PWCFlow(num_levels = num_levels, num_channels_upsampled_context=32,
                                               use_cost_volume=use_cost_volume, use_feature_warp=True,
                                               action_channels=action_channels)

    def forward(self, img1, img2, actions=None):
        """Compute the flow between image pairs
        :param img1: [BCHW] image 1 batch
        :param img2: [BCHW] image 2 batch
        :param actions: None or [BC] self-action for each pair, C == self.action_channels
        :return: [B2HW] flow tensor for each image pair in batch, feature pyramid list for img pairs
        """
        fp1 = self._pyramid(img1)
        fp2 = self._pyramid(img2)
        flow = self._flow_model(fp1, fp2, actions)
        return flow, fp1, fp2

    def compute_loss(self, img1, img2, features1, features2, flows,ncc=None):
        f1 = [img1] + features1
        f2 = [img2] + features2

        warps = [uflow_utils.flow_to_warp(f) for f in flows]
        warped_f2 = [uflow_utils.resample(f, w) for (f, w) in zip(f2, warps)]

        loss = uflow_utils.compute_all_loss(f1, warped_f2, flows,ncc)
        return loss