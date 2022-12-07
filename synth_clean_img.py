import os
import numpy as np
import pandas as pd
from utils.planetaryimageEDR import PDS3ImageEDR
from parfor import parfor

from dataset_creation.noisy_imgs import (
    non_linearity,
    flatfield,
)

NOISY_PATCH_PATH = '/media/panlab/CHARVIHDD/PhotonNet/NAC_L/test/noisy'
CLEAN_PATCH_DIR = '/media/panlab/CHARVIHDD/PhotonNet/NAC_L/test/clean'
DESTRIPE_PATCH_PATH = '/media/panlab/CHARVIHDD/PhotonNet/NAC_L/test/destripe_output'
SYNTH_CLEAN_PATH = '/media/panlab/CHARVIHDD/PhotonNet/NAC_L/test/result_destripe_noise'
'''
NOISY_PATCH_PATH: source path to noisy image patches
DESTRIPE_PATCH_PATH: source path to destripe output patches
SYNTH_CLEAN_PATH: destination path to save synthetic clean images
'''


def get_synth_clean_img(noisy_patch, destripe_patch):
    J = noisy_patch - destripe_patch
    J[destripe_patch > noisy_patch] = 0
    J = non_linearity(
        image=J,
        mode_and_camera='summed_L',
        forward=False
    )
    J = flatfield(
        image=J,
        mode_and_camera='summed_L',
        forward=False
    )

    return J

def main():
    noisy_patch_files = os.listdir(NOISY_PATCH_PATH)
    destripe_patch_files = os.listdir(DESTRIPE_PATCH_PATH)

    # for i in range(len(noisy_patch_files)):
    @parfor(range(len(noisy_patch_files)))
    def generate(i):
        assert noisy_patch_files[i] == destripe_patch_files[i]
        with open(os.path.join(DESTRIPE_PATCH_PATH, destripe_patch_files[i]), 'rb') as f2:
            destripe_img = np.fromfile(f2, dtype=np.uint16).reshape(52224, 2532)
        noisy_img = PDS3ImageEDR.open(os.path.join(NOISY_PATCH_PATH, noisy_patch_files[i])).image

        J = get_synth_clean_img(
            noisy_img,
            destripe_img,
        )

        with open(os.path.join(SYNTH_CLEAN_PATH, 'synth_{}'.format(noisy_patch_files[i])), 'wb') as f3:
            f3.write(bytes(J))
        f3.close()


def evaluate_L1():
    clean_patch_files = os.listdir(CLEAN_PATCH_DIR)
    destripe_patch_files = os.listdir(DESTRIPE_PATCH_PATH)
    L1 = 0
    for i in range(len(clean_patch_files)):
        with open(os.path.join(DESTRIPE_PATCH_PATH, destripe_patch_files[i]), 'rb') as f2:
            destripe_img = np.fromfile(f2, dtype=np.uint16)
        clean_img = PDS3ImageEDR.open(os.path.join(NOISY_PATCH_PATH, clean_patch_files[i])).image
        image_L1 = np.linalg.norm(clean_img.flatten().astype(float) - destripe_img.astype(float), ord=1) / \
                   destripe_img.size
        L1 += image_L1
        print(image_L1)

    return L1 / len(clean_patch_files)

if __name__ == '__main__':
    print(evaluate_L1())
        