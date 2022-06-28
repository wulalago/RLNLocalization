# ========================================
# Perform alignment based on Prior Library
# ========================================

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import center_of_mass
from medpy.metric import dc
from torchvision import transforms as T
from op.data_op import load_list
from dipy.align import imaffine
from dipy.align import transforms

TEXT_PATH = "Text"
VIEW = ['left']
VISUAL = 'Visual'
PRIOR_DATA = 'PRIOR'


def double_align(tissues_mask, segmentations_mask, rln_and_tissues_mask):
    identity = np.eye(3)
    c_of_mass = imaffine.transform_centers_of_mass(segmentations_mask, identity, tissues_mask, identity)

    n_bins = 32
    sampling_prop = None
    metric = imaffine.MutualInformationMetric(n_bins, sampling_prop)
    level_iter = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affine_reg = imaffine.AffineRegistration(metric=metric, level_iters=level_iter, sigmas=sigmas, factors=factors,
                                             verbosity=0)

    transform = transforms.TranslationTransform2D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affine_reg.optimize(segmentations_mask, tissues_mask, transform, params0, identity, identity,
                                      starting_affine=starting_affine)

    # transformed_img = translation.transform(img, interpolation='linear')
    transformed_tissues_mask = translation.transform(tissues_mask, interpolation='nearest')
    transformed_rln_and_tissues_mask = translation.transform(rln_and_tissues_mask, interpolation='nearest')

    transformed_tissues_mask = transformed_tissues_mask / 50
    transformed_tissues_mask = transformed_tissues_mask.astype(np.int32)
    transformed_tissues_mask *= 50

    transformed_rln_and_tissues_mask = transformed_rln_and_tissues_mask / 50
    transformed_rln_and_tissues_mask = transformed_rln_and_tissues_mask.astype(np.int32)
    transformed_rln_and_tissues_mask *= 50

    return transformed_tissues_mask, transformed_rln_and_tissues_mask


if __name__ == '__main__':
    train_list, val_list, test_list = load_list(TEXT_PATH, VIEW)
    target_list = test_list
    source_list = val_list + train_list
    os.makedirs(VISUAL, exist_ok=True)
    os.makedirs(PRIOR_DATA, exist_ok=True)
    mask_transform = T.Compose([
            T.Resize((256, 256), Image.NEAREST),
        ])

    target_dice_list = []
    for target_path in target_list[:10]:
        patient_id, view_type, item_id = target_path.split('\\')[1:]
        segmentations_path = 'Results/{}-{}-{}.png'.format(patient_id, view_type, item_id)
        seg_msk = Image.open(segmentations_path).convert('L')
        seg_msk_arr = np.array(seg_msk, dtype=np.int32)

        target_item_dice_list = []
        prior_item_center_list = []

        for source_path in tqdm(source_list):
            temp_list = []
            for idx, mask_item in enumerate(["CCA", "thyroid", "trachea", "RLN"]):
                msk = Image.open(os.path.join(source_path, "MASK", "{}.jpg".format(mask_item)))
                msk = mask_transform(msk)
                msk = np.array(msk) / 255 * (idx + 1)
                temp_list.append(msk)

            tissues_msk_arr = np.stack(temp_list[:-1], axis=0)
            tissues_msk_arr = np.max(tissues_msk_arr, axis=0) * 50
            tissues_msk_arr = tissues_msk_arr.astype(np.int32)

            rln_msk_arr = np.stack(temp_list, axis=0)
            rln_msk_arr = np.max(rln_msk_arr, axis=0) * 50
            rln_msk_arr = rln_msk_arr.astype(np.int32)

            aligned_tissues_msk, aligned_rln_msk = double_align(tissues_msk_arr, seg_msk_arr, rln_msk_arr)

            dice_aligned_seg = dc(aligned_tissues_msk, seg_msk_arr)
            target_item_dice_list.append(dice_aligned_seg)

            prior_rln_msk = aligned_rln_msk == 200
            prior_rln_msk = prior_rln_msk.astype(np.int32)
            center_coord = center_of_mass(prior_rln_msk)
            prior_item_center_list.append([center_coord[0], center_coord[1], dice_aligned_seg])

        target_item_dice_list.sort(reverse=True)
        target_dice_list.append(target_item_dice_list)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(seg_msk_arr)
        for temp in prior_item_center_list:
            if temp[2] > 0.85:
                s = 15
            elif 0.75 > temp[2] > 0.85:
                s = 5
            else:
                s = 1
            plt.scatter(x=temp[1], y=temp[0], s=s, alpha=0.3, c="red")

        plt.subplot(1, 2, 2)
        plt.plot(target_item_dice_list)
        plt.title('Sorted Dice score of each aligned prior mask to the segmentation')
        plt.xlabel('Subj Id')
        plt.ylabel('Dice')
        # plt.show()
        plt.savefig(os.path.join(VISUAL, '{}-{}-{}.png'.format(patient_id, view_type, item_id)))
        plt.close()

        prior_item_center_arr = np.array(prior_item_center_list)
        np.save(os.path.join(PRIOR_DATA,  '{}-{}-{}.npy'.format(patient_id, view_type, item_id)), prior_item_center_arr)

        # x = np.arange(1, len(target_item_dice_list) + 1)

    plt.figure()
    for temp in target_dice_list:
        plt.plot(temp)
    plt.show()

