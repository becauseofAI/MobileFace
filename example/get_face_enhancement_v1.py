import os, sys
import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '../MobileFace_Enhancement/')
from mobileface_enhancement import MobileFaceEnhance

def parse_args():
    parser = argparse.ArgumentParser(description='Test MobileFace Makeup.')
    parser.add_argument('--image-dir', type=str, default='./light',
                        help='Test images directory.')
    parser.add_argument('--result-dir', type=str, default='./light_result',
                        help='Result images directory.')
    parser.add_argument('--dark-th', type=int, default=80,
                        help='Black pixel threshold whith typical values from 50 to 100.')
    parser.add_argument('--bright-th', type=int, default=200,
                        help='White pixel threshold whith typical values from 180 to 220.')
    parser.add_argument('--dark-shift', type=float, default=0.4,
                        help='Gamma shift value for gamma correction to brighten the face. \
                            The typical values are from 0.3 to 0.5.')
    parser.add_argument('--bright-shift', type=float, default=2.5,
                        help='Gamma shift value for gamma correction to darken the face. \
                            The typical values are from 2.0 to 3.0.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args() 
    enhance_tool = MobileFaceEnhance()
    img_list = os.listdir(args.image_dir)
    for img_name in img_list:
        im_path = os.path.join(args.image_dir, img_name)
        img = cv2.imread(im_path)
        gamma, hist = enhance_tool.hist_statistic(img, 
                                                  dark_th = args.dark_th, 
                                                  bright_th = args.bright_th, 
                                                  dark_shift = args.dark_shift, 
                                                  bright_shift = args.bright_shift)
        img_gamma = enhance_tool.gamma_trans(img, gamma)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        rst_path = os.path.join(args.result_dir, 'rst_debug' + img_name)
        cv2.imwrite(rst_path, img_gamma)

        img_stack = np.hstack((img, img_gamma))

        plt.figure(figsize=(5, 5))
        ax1 = plt.subplot2grid((2,1),(0, 0))
        ax1.set_title('Grayscale Histogram')
        ax1.set_xlabel("Bins")
        ax1.set_ylabel("Num of Pixels")
        ax1.plot(hist)
        ax1.set_xlim([0, 256])

        ax1 = plt.subplot2grid((2, 1), (1, 0), colspan=3, rowspan=1)
        ax1.set_title('Enhance Comparison')
        ax1.imshow(img_stack[:,:,::-1])

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()