import os
import numpy as np
import pandas as pd

from dataset_creation.noisy_imgs import (
    non_linearity,
    flatfield,
)

NOISY_PATCH_PATH = ''
DESTRIPE_PATCH_PATH = ''
SYNTH_CLEAN_PATH = ''
'''
NOISY_PATCH_PATH: source path to noisy image patches
DESTRIPE_PATCH_PATH: source path to destripe output patches
SYNTH_CLEAN_PATH: destination path to save synthetic clean images
'''


def get_synth_clean_img(noisy_patch, destripe_patch):
    J = noisy_patch - destripe_patch
    J = non_linearity(
        image=J,
        mode_and_camera='summed L',
        forward=False,
    )
    J = flatfield(
        image=J,
        mode_and_camera='summed L',
    )

    return J

def main():
    noisy_patch_files = os.listdir(NOISY_PATCH_PATH)
    destripe_patch_files = os.listdir(DESTRIPE_PATCH_PATH)

    for i in range(len(noisy_patch_files)):
        assert noisy_patch_files[i] == destripe_patch_files[i]

        with open(os.path.join(NOISY_PATCH_PATH, noisy_patch_files[i]), 'rb') as f1:
            noisy_patch = np.fromfile(f1, dtype=np.uint16).reshape(256, 256)
        f1.close()

        with open(os.path.join(DESTRIPE_PATCH_PATH, destripe_patch_files[i]), 'rb') as f2:
            destripe_patch = np.fromfile(f2, dtype=np.uint16).reshape(256, 256)
        f2.close()


        J = get_synth_clean_img(
            noisy_patch[i], 
            destripe_patch[i]
        )

        with open(os.path.join(SYNTH_CLEAN_PATH, 'synth_{}'.format(noisy_patch_files.split('_')[-1])), 'wb') as f3:
            f3.write(bytes(J))
        f3.close()

if __name__ == '__main__':
    main()
        