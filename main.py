import os

from dataset_creation import noisy_imgs

def main():
    # source_dir = '/media/panlab/EXTERNALHDD/bright_summed/'
    # # generate_crop_list(source_dir, 1e5, destination_npy_file='crop_list_R.npy')
    # destination_dir = '/media/panlab/CHARVIHDD/SYDE671/DestripeNet/'
    # calib_dir = '/media/panlab/EXTERNALHDD/dark_summed/'
    # for sub_dir in os.listdir(source_dir):
    #     crop_list = np.load('crop_list_' + sub_dir[-1] + '.npy', allow_pickle=True)
    #     generate_noisy_img_pairs(source_dir + sub_dir, destination_dir + sub_dir,
    #                              crop_list, calib_dir + '/' + sub_dir + '/', 'summed_' + sub_dir[-1])
    dark_calibration = '/media/panlab/EXTERNALHDD/dark_summed/'
    destination_dir = dark_calibration
    for sub_dir in os.listdir(dark_calibration):
        noisy_imgs.generate_destripe_data(os.path.join(dark_calibration, sub_dir),
                                          os.path.join(destination_dir, sub_dir))


if __name__ == '__main__':
    main()
