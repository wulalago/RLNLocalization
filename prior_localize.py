import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms as T
from scipy.ndimage import center_of_mass

PRIOR_PATH = 'PRIOR_right'
SAVE_PATH = 'Prior_Results_right'
GT_PATH = '../Dataset/Data'

mask_transform = T.Compose([
    T.Resize((256, 256), Image.NEAREST),
])


if __name__ == '__main__':
    os.makedirs(SAVE_PATH, exist_ok=True)
    prior_list = os.listdir(PRIOR_PATH)
    error_mean = 0.
    error_weighted_mean = 0.
    count = 0
    error_arr = []
    for prior_name in prior_list:
        prior_data = np.load(os.path.join(PRIOR_PATH, prior_name))

        nan_check = ~np.isnan(prior_data).any(axis=1)
        prior_data = prior_data[nan_check, :]

        dice_threshold = 0.8
        failed_check = prior_data[:, 2] > dice_threshold
        prior_data = prior_data[failed_check, :]

        K = 10
        top_orders = np.argsort(prior_data[:, 2])
        top_orders = np.flipud(top_orders)
        top_orders = top_orders[:K]
        prior_data = prior_data[top_orders, :]

        prior_name = prior_name.split('.npy')[0]
        patient_id, view_type_1, view_type_2, item_id = prior_name.split('-')
        view_type = view_type_1 + '-' + view_type_2

        temp_list = []
        for idx, mask_item in enumerate(["CCA", "thyroid", "trachea", "RLN"]):
            msk = Image.open(os.path.join(GT_PATH, patient_id, view_type, item_id, "MASK", "{}.jpg".format(mask_item)))
            msk = mask_transform(msk)
            msk = np.array(msk) / 255 * (idx + 1)
            temp_list.append(msk)

        gt_msk_arr = np.stack(temp_list, axis=0)
        gt_msk_arr = np.max(gt_msk_arr, axis=0) * 50
        gt_msk_arr = gt_msk_arr.astype(np.int32)

        gt_rln_msk = gt_msk_arr == 200
        gt_rln_msk = gt_rln_msk.astype(np.int32)
        gt_rln_coord = list(center_of_mass(gt_rln_msk))

        mean_rln_coord = np.mean(prior_data, axis=0)

        weight = np.exp(prior_data[:, 2])
        weight /= weight.sum()
        weighted_mean_rln_coord = np.sum(prior_data[:, :2] * weight[:, np.newaxis], axis=0)

        dist_mean_gt = np.sqrt(np.sum((mean_rln_coord[:2] - gt_rln_coord) ** 2))
        dist_weighted_mean_gt = np.sqrt(np.sum((weighted_mean_rln_coord - gt_rln_coord) ** 2))

        plt.figure()
        plt.imshow(gt_msk_arr)
        for temp in prior_data:
            plt.scatter(x=temp[1], y=temp[0], s=1, alpha=0.1, c="red")

        plt.scatter(x=gt_rln_coord[1], y=gt_rln_coord[0], s=5, c='blue', label='GT')
        plt.scatter(x=mean_rln_coord[1], y=mean_rln_coord[0], s=5, c='whitesmoke', label='Mean')
        plt.scatter(x=weighted_mean_rln_coord[1], y=weighted_mean_rln_coord[0], s=5, c='cyan', label='Weighted Mean')
        plt.legend()
        plt.savefig(os.path.join(SAVE_PATH,  '{}-{}-{}.png'.format(patient_id, view_type, item_id)))
        plt.close()

        print(prior_name, dist_mean_gt, dist_weighted_mean_gt, gt_rln_coord, mean_rln_coord, weighted_mean_rln_coord)
        error_mean += dist_mean_gt
        error_weighted_mean += dist_weighted_mean_gt

        if np.isnan(np.sum(dist_mean_gt)):
            error_arr.append(dist_mean_gt)
        else:
            error_arr.append(dist_mean_gt)
        count += 1

        save_list = {'Mean': mean_rln_coord,
                     'Weighted_Mean': weighted_mean_rln_coord,
                     'K': K,
                     'Dice_Threshold': dice_threshold}
        np.save(os.path.join(SAVE_PATH,  '{}-{}-{}.npy'.format(patient_id, view_type, item_id)), save_list)

    print(np.mean(error_arr), np.std(error_arr))
    print(np.mean(np.array(error_arr) < 15))
    print("Avg", error_mean/count, error_weighted_mean/count)


