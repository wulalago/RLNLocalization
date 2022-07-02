import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from medpy.metric import dc
from dipy.align import imaffine
from dipy.align import transforms


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def set_device(cuda):
    """
    Set the torch gpu device
    TODO: parallel setup is requested
    ----------------------------
    Parameters:
        cuda: [int] id of the used GPU, where -1 is "cpu"
    Return:
        torch device
    ----------------------------
    """
    assert isinstance(cuda, int)

    if cuda == -1 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(cuda))

    return device


class Plotter(object):
    """
    Plot the loss/metric curves
    """
    def __init__(self, send_path):
        """
        send_path: [string] path to save the figures
        """
        self.send_path = send_path
        self.buffer = dict()

    def update(self, logs):
        """
        logs: [dict] metric dict that to be plot
        """
        for key in logs.keys():
            if key not in self.buffer.keys():
                self.buffer[key] = []
            self.buffer[key].append(logs[key])

    def send(self):
        """
        function to plot the curve
        """
        for key in self.buffer.keys():
            plt.figure()
            plt.plot(self.buffer[key])
            plt.title(key)
            plt.xlabel("epoch")
            plt.savefig(os.path.join(self.send_path, key+".png"))
            plt.close()


class Recorder(object):
    """
    record the metric and return the statistic results
    """
    def __init__(self, keys):
        """
        keys: [list] variables' name to be saved
        """
        self.data = dict()
        self.keys = keys
        for key in keys:
            self.data[key] = []

    def update(self, item):
        """
        item: [dict] data dict to update the buffer, the keys should be consistent
        """
        for key in item.keys():
            self.data[key].append(item[key])

    def reset(self, keys=None):
        """
        keys: [list] variables to be cleared in the buffer
        """
        if keys is None:
            keys = self.data.keys()
        for key in keys:
            self.data[key] = []

    def call(self, key, return_std=False):
        """
        key: [string] variable to be calculated for the statistical results
        return_std: [bool] option to return variance
        """
        arr = np.array(self.data[key])
        if return_std:
            return np.mean(arr), np.std(arr)
        else:
            return np.mean(arr)


def array2tensor(array, dtype="float32"):
    """
    transfer the numpy array to the torch tensor
    TODO: more dtype is requested
    ----------------------------
    Parameters:
        array: [numpy.array] array to be transferred
        dtype: [string] type of the tensor, current only support Float32 and Int64
    Return:
        torch tensor
    ----------------------------
    """
    tensor = torch.from_numpy(array)
    if dtype == "float32":
        return tensor.float()
    elif dtype == "int64":
        return tensor.long()
    else:
        raise NameError("Currently only support Float32 and Int64")


def tensor2array(tensor, squeeze=False):
    """
    transfer the torch tensor to the numpy array
    ----------------------------
    Parameters:
        tensor: [torch.Tensor] tensor to be transferred
        squeeze: [bool] option for squeeze the tensor
    Return:
        numpy array
    ----------------------------
    """
    if squeeze:
        tensor = tensor.squeeze()
    return tensor.cpu().detach().numpy()


def procrustes_analysis(reference_mask, mask):
    identity = np.eye(3)
    c_of_mass = imaffine.transform_centers_of_mass(reference_mask, identity, mask, identity)

    n_bins = 32
    sampling_prop = None
    metric = imaffine.MutualInformationMetric(n_bins, sampling_prop)
    level_iter = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affine_reg = imaffine.AffineRegistration(metric=metric, level_iters=level_iter, sigmas=sigmas, factors=factors)

    transform = transforms.TranslationTransform2D()
    params0 = None
    translation = affine_reg.optimize(reference_mask, mask, transform, params0, identity, identity,
                                      starting_affine=c_of_mass.affine)

    transform = transforms.RigidTransform2D()
    rigid = affine_reg.optimize(reference_mask, mask, transform, params0, identity, identity, starting_affine=translation.affine)


    # transformed_img = rigid.transform(img, interpolation='linear')
    transformed_mask = rigid.transform(mask, interpolation='nearest')
    transformed_mask = transformed_mask / 50
    transformed_mask = transformed_mask.astype(np.int32)
    transformed_mask *= 50

    print(set(list(transformed_mask.reshape(-1))))
    return



