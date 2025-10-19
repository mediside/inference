import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import load_arr

def crop_area(img, mask):
    # обрезка по непустым областям маски
    for idx in range(mask.shape[1]):
        if mask[:, idx].sum() != 0:
            l_idx = idx
            break
    for idx in range(mask.shape[1] - 1, -1, -1):
        if mask[:, idx].sum() != 0:
            r_idx = idx
            break
    img = img[:, l_idx:r_idx + 1]
    mask = mask[:, l_idx:r_idx + 1]

    for idx in range(mask.shape[0]):
        if mask[idx, :].sum() != 0:
            l_idx = idx
            break
    for idx in range(mask.shape[0] - 1, -1, -1):
        if mask[idx, :].sum() != 0:
            r_idx = idx
            break
    img = img[l_idx:r_idx + 1, :]
    mask = mask[l_idx:r_idx + 1, :]
    return img, mask


def normalize(volume, max=0, min=-800):
    volume = np.clip(volume, min, max)
    volume = (volume - min) / (max - min)
    return volume


def make_projections(path_ct, path_mask, save_dir, iter_erode=2):
    """
    Делает MIP-проекции левого и правого лёгкого из КТ и сохраняет PNG.
    Работает и с .nii.gz, и с DICOM-папками.
    """

    os.makedirs(save_dir, exist_ok=True)
    patient_name = os.path.basename(path_ct).replace('.nii.gz', '')

    # загрузка КТ
    # ct_img, _ = loader(path_ct)
    # ct_sitk = sitk.GetImageFromArray(np.asarray(ct_img))
    # ct_data = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
    ct_data = load_arr(path_ct)
    ct_data[ct_data < -1024] = -1024
    ct_data = normalize(ct_data)

    # загрузка маски
    # mask_img, _ = loader(path_mask)
    # mask_sitk = sitk.GetImageFromArray(np.asarray(mask_img))
    # mask_data = sitk.GetArrayFromImage(mask_sitk)
    mask_data = load_arr(path_mask)

    # эрозия маски
    kernel = np.ones((3, 3), np.uint8)
    for idx in range(mask_data.shape[0]):
        for _ in range(iter_erode):
            try:
                mask_data[idx, :, :] = cv2.erode(mask_data[idx, :, :], kernel, iterations=1)
            except:
                pass

    # разделение на левое/правое
    leftmask = (mask_data == 2).astype(np.uint8)
    rightmask = (mask_data == 1).astype(np.uint8)

    left = ct_data.copy()
    right = ct_data.copy()
    left[leftmask != 1] = left.min()
    right[rightmask != 1] = right.min()

    projection_names = ['r_a', 'r_c', 'r_s', 'l_a', 'l_c', 'l_s']
    projection_list, projection_masklist = [], []

    # правое лёгкое
    for idx in range(3):
        projection_list.append(np.max(right, axis=idx)[::-1, :])
        mask = np.max(rightmask, axis=idx)[::-1, :]
        projection_masklist.append((mask > 0).astype(np.uint8))

    # левое лёгкое
    for idx in range(3):
        projection_list.append(np.max(left, axis=idx)[::-1, :])
        mask = np.max(leftmask, axis=idx)[::-1, :]
        projection_masklist.append((mask > 0).astype(np.uint8))

    # сохранение
    for pos, img, mask in zip(projection_names, projection_list, projection_masklist):
        img, mask = crop_area(img, mask)
        plt.imsave(os.path.join(save_dir, f"{patient_name}_{pos}.png"), img, cmap='gray')
        plt.imsave(os.path.join(save_dir, f"{patient_name}_{pos}_mask.png"), mask, cmap='gray')
