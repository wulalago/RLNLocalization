import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import center_of_mass
from random import uniform
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import crop, to_tensor


def load_list(text_path, view):

    def _read_text(path):
        data = open(path, 'r').readlines()
        data = [item.replace('\n', '') for item in data]
        return data

    train_list, val_list, test_list = [], [], []
    if 'left' in view:
        train_list += _read_text(os.path.join(text_path, "train_L-RLN.txt"))
        val_list += _read_text(os.path.join(text_path, "val_L-RLN.txt"))
        test_list += _read_text(os.path.join(text_path, "test_L-RLN.txt"))
    if 'right' in view:
        train_list += _read_text(os.path.join(text_path, "train_R-RLN.txt"))
        val_list += _read_text(os.path.join(text_path, "val_R-RLN.txt"))
        test_list += _read_text(os.path.join(text_path, "test_R-RLN.txt"))

    return train_list, val_list, test_list


class RLNDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.img_transform = T.Compose([
            T.Resize((256, 256), Image.BILINEAR),
            T.ToTensor(),
        ])
        self.mask_transform = T.Compose([
            T.Resize((256, 256), Image.NEAREST),
        ])

    def __getitem__(self, item):
        item_path = self.data_list[item].replace('\\', '/')
        img_name = os.listdir(os.path.join(item_path, "IMG"))[0]
        img_path = os.path.join(item_path, "IMG", img_name)

        item_name = '{}-{}-{}'.format(*item_path.split('/')[3:])

        img = Image.open(img_path).convert('L')
        img_tensor = self.img_transform(img)
        msk_list = []
        for idx, mask_item in enumerate(["CCA", "thyroid", "trachea"]):
            msk = Image.open(os.path.join(item_path, "MASK", "{}.jpg".format(mask_item)))
            msk = self.mask_transform(msk)
            msk = np.array(msk) / 255 * (idx+1)
            msk_list.append(torch.from_numpy(msk).long())
        msk_tensor = torch.stack(msk_list, dim=0)
        msk_tensor = torch.max(msk_tensor, dim=0)[0]
        return img_tensor, msk_tensor, item_name

    def __len__(self):
        return len(self.data_list)


class RLNRefineDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.img_transform = T.Compose([
            T.Resize((256, 256), Image.BILINEAR),
            # T.ToTensor(),
        ])
        self.mask_transform = T.Compose([
            T.Resize((256, 256), Image.NEAREST),
        ])

    def __getitem__(self, item):
        item_path = self.data_list[item].replace('\\', '/')
        img_name = os.listdir(os.path.join(item_path, "IMG"))[0]
        img_path = os.path.join(item_path, "IMG", img_name)

        item_name = '{}-{}-{}'.format(*item_path.split('/')[3:])

        img = Image.open(img_path).convert('L')
        img = self.img_transform(img)

        rln_path = os.path.join(item_path, "MASK", "RLN.jpg")
        rln_msk = Image.open(rln_path).convert('L')
        rln_msk = self.mask_transform(rln_msk)
        rln_msk_arr = np.array(rln_msk)
        rln_coord = list(center_of_mass(rln_msk_arr))

        # ==================== Single Crop =================================== #
        # top, left = rln_coord[0] - 32, rln_coord[1] - 32   # patch size is 64
        #
        # cropped_center_h, cropped_center_w = 32, 32
        # height, width = 64, 64
        #
        # random_shift_h, random_shift_w = uniform(-15, 15), uniform(-15, 15)
        #
        # top += random_shift_h
        # left += random_shift_w
        #
        # cropped_center_h -= random_shift_h
        # cropped_center_w -= random_shift_w
        #
        # cropped_img = crop(img, top=top, left=left, height=height, width=width)
        # ==================== Single Crop =================================== #

        # ==================== Multi Crop =================================== #
        cropped_center_h, cropped_center_w = 32, 32

        random_shift_h, random_shift_w = uniform(-20, 20), uniform(-20, 20)

        cropped_center_h -= random_shift_h
        cropped_center_w -= random_shift_w

        cropped_img_s = crop(
            img,
            top=rln_coord[0] + random_shift_h - 12,
            left=rln_coord[1] + random_shift_w - 12,
            height=24, width=24
        )
        # cropped_img_m = crop(
        #     img,
        #     top=rln_coord[0] + random_shift_h - 16,
        #     left=rln_coord[1] + random_shift_w - 16,
        #     height=32, width=32
        # )
        cropped_img_l = crop(
            img,
            top=rln_coord[0] + random_shift_h - 32,
            left=rln_coord[1] + random_shift_w - 32,
            height=64, width=64
        )

        # ==================== Multi Crop =================================== #

        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(cropped_img, cmap='gray')
        # plt.scatter(x=cropped_center_w, y=cropped_center_h, s=5, c='cyan')
        # plt.subplot(1, 2, 2)
        # plt.imshow(img, cmap='gray')
        # plt.scatter(x=rln_coord[1], y=rln_coord[0], s=5, c='cyan')
        # plt.show()

        # img_tensor = to_tensor(img.convert('RGB'))
        cropped_img_s_tensor = to_tensor(cropped_img_s)
        # cropped_img_m_tensor = to_tensor(cropped_img_m)
        cropped_img_l_tensor = to_tensor(cropped_img_l)

        center_coord = torch.from_numpy(np.array([cropped_center_h, cropped_center_w])).float()
        return [cropped_img_s_tensor, cropped_img_l_tensor], center_coord, item_name

    def __len__(self):
        return len(self.data_list)


class RLNRriorDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.img_transform = T.Compose([
            T.Resize((256, 256), Image.BILINEAR),
            # T.ToTensor(),
        ])
        self.mask_transform = T.Compose([
            T.Resize((256, 256), Image.NEAREST),
        ])

    def __getitem__(self, item):
        item_path = self.data_list[item].replace('\\', '/')
        img_name = os.listdir(os.path.join(item_path, "IMG"))[0]
        img_path = os.path.join(item_path, "IMG", img_name)

        item_name = '{}-{}-{}'.format(*item_path.split('/')[3:])

        prior_result_path = os.path.join('Prior_Results', '{}-{}-{}.npy'.format(*item_path.split('/')[3:]))
        prior_result = np.load(prior_result_path, allow_pickle=True).item()
        init_coord = prior_result['Weighted_Mean']

        img = Image.open(img_path).convert('L')
        img = self.img_transform(img)

        rln_path = os.path.join(item_path, "MASK", "RLN.jpg")
        rln_msk = Image.open(rln_path).convert('L')
        rln_msk = self.mask_transform(rln_msk)
        rln_msk_arr = np.array(rln_msk)
        rln_coord = list(center_of_mass(rln_msk_arr))
        # ==================== Single Crop =================================== #
        # top, left = init_coord[0] - 32, init_coord[1] - 32   # patch size is 64
        # height, width = 64, 64
        # cropped_rln_coord = [rln_coord[0] - top, rln_coord[1] - left]
        # cropped_img = crop(img, top=top, left=left, height=height, width=width)
        # ==================== Single Crop =================================== #


        # ==================== Multi Crop =================================== #
        top, left = init_coord[0] - 32, init_coord[1] - 32
        cropped_rln_coord = [rln_coord[0] - top, rln_coord[1] - left]

        cropped_img_s = crop(
            img,
            top=init_coord[0] - 12,
            left=init_coord[1] - 12,
            height=24, width=24
        )
        # cropped_img_m = crop(
        #     img,
        #     top=init_coord[0] - 16,
        #     left=init_coord[1] - 16,
        #     height=32, width=32
        # )

        cropped_img_l = crop(
            img,
            top=init_coord[0] - 32,
            left=init_coord[1] - 32,
            height=64, width=64
        )

        # ==================== Multi Crop =================================== #


        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(cropped_img, cmap='gray')
        # plt.scatter(x=cropped_rln_coord[1], y=cropped_rln_coord[0], s=5, c='cyan')
        # plt.scatter(x=32, y=32, s=5, c='red')
        # plt.subplot(1, 2, 2)
        # plt.imshow(img, cmap='gray')
        # plt.scatter(x=rln_coord[1], y=rln_coord[0], s=5, c='cyan')
        # plt.show()
        img_tensor = to_tensor(img.convert('RGB'))
        # cropped_img_tensor = to_tensor(cropped_img)

        cropped_img_s_tensor = to_tensor(cropped_img_s)
        # cropped_img_m_tensor = to_tensor(cropped_img_m)
        cropped_img_l_tensor = to_tensor(cropped_img_l)

        center_coord = torch.from_numpy(np.array(cropped_rln_coord))
        move_coord = torch.from_numpy(np.array([top, left]))
        return img_tensor, [cropped_img_s_tensor, cropped_img_l_tensor], center_coord, move_coord, item_name

    def __len__(self):
        return len(self.data_list)