from algorithm.utils.data.tf.dataimage import DataImage
from algorithm.deep.tf.neural_networks import ResNet1, ResNet2__0, ResNet2__1, ResNet2__0__1, ResNet2__1__1
from algorithm.utils.params.tf.dr import DomainRandomization_parameters
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import argparse
import tensorflow as tf
import keras
import nevergrad as ng
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('--data_path', type=str, help='Path of the directory in which input data are stored\n(default: Input\dataset)', default="Input\\dataset")
parser.add_argument('--split', type=str, help="Percentage to split the dataset into train/validation sets. It can be a float between 0 and 1 (representing the percentage), or 'auto' (to use the standard split: 0.2) or 'train only' (to use all the dataset for training. Use this option if only training is needed).\n(default: 'train only')", default="train only")
parser.add_argument('--transform', type=str, help="Transformations to be applied to images of the dataset.\n(default: None, i.e.: no transformations are appied.)", default=None)
parser.add_argument('--normalize', action='store_true', help='Normalize option to be applied to images of the dataset.\n(default: True)', default=True)
nn_desc  = """
Neural Networks:

- ResNet 1: very fast training, potential low performances.
- ResNet 2.0: slowest training, able to reach the best performances.
- ResNet 2.1: fast training, potential low performances, generally better than ResNet 1.
- ResNet 2.0.1: deeper version of ResNet 2.0, useful for large datasets with many classes.
- ResNet 2.1.1: deeper version of ResNet 2.1, useful for large datasets with many classes.
"""
standard_norm = """
Supported Color Modes and Standard Normalization Parameters:

1. RGB:
    - Mean: [0.00392157, 0.00392157, 0.00392157]
    - Standard Deviation (Std): [0.229, 0.224, 0.225]

2. Grayscale:
    - Mean: [0.5]
    - Std: [0.5]

3. RGBA:
    - Mean: [0.485, 0.456, 0.406, 0.0]
    - Std: [0.229, 0.224, 0.225, 1.0]

Note: 
- The mean and standard deviation (std) values are used for normalization in the respective color modes.
- RGB and RGBA mean values are scaled to the range [0, 1] by dividing by 255.
"""
standard_res = """
Standard Resize Parameters:

- Height: 128
- Width: 128
"""
image_params = """
Image Parameters:

- brightness
- contrast
- horizontally flip
- vertically flip
- hue
- jpeg quality
- saturation
"""
dr__mode  = """
Domain randomization, mode. Distributions:

- multivariate normal
- univariate normal
- uniform
- triangular
"""
standard_facts = """
Domain randomization, standard factor parameters.

Probability that for a single (batch of) image the corresponding parameter is randomized:
- brightness, 0.9
- contrast, 0.9
- horizontally flip, 0.9
- vertically flip, 0.9
- hue, 0.9
- jpeg quality, 0.9
- saturation, 0.9
"""
standard_pars = """
Domain randomization, standard distribution parameters.

The meaning of the parameter depends on the distribution(s) considered. If univariate distributions are used, the order reflects the corresponding image parameter to be randomized:
- Univariate uniform distribution, 
\tU(lower_brightness, upper_brightness);  U(lower_contrast, upper_contrast);  U(lower_horizontal_flip, upper_horizontal_flip);  U(lower_vertical_flip, upper_vertical_flip);  U(lower_hue, upper_hue);  U(lower_jpeg_quality, upper_jpeg_quality);  U(lower_saturation, upper_saturation)
\tstandard distribution parameters, U(-0.2, 0.2);  U(0, 2.5);  U(0, 1);  U(0, 1);  U(-0.2, 0.2);  U(20, 100);  U(0, 2)
\t\tLinearized version: Given lowers=[-0.2, 0, 0, 0, -0.2, 20, 0], uppers=[0.2, 2.5, 1, 1, 0.2, 100, 2] -> -0.2 0 0 0 -0.2 20 0 0.2 2.5 1 1 0.2 100 2
- Univariate triangular distribution, 
\tTr(lower_brightness, mode_brightness, upper_brightness);  Tr(lower_contrast, mode_contrast, upper_contrast);  Tr(lower_horizontal_flip, mode_horizontal_flip, upper_horizontal_flip);  Tr(lower_vertical_flip, mode_vertical_flip, upper_vertical_flip);  Tr(lower_hue, mode_hue, upper_hue);  Tr(lower_jpeg_quality, mode_jpeg_quality, upper_jpeg_quality);  Tr(lower_saturation, mode_saturation, upper_saturation)
\tstandard distribution parameters, Tr(-0.2, 0, 0.2);  Tr(0, 1.25, 2.5);  Tr(0, 0.5, 1);  Tr(0, 0.5, 1);  Tr(-0.2, 0, 0.2);  Tr(20, 60, 100);  Tr(0, 1, 2)
\t\tLinearized version: Given lowers=[-0.2, 0, 0, 0, -0.2, 20, 0], modes=[0, 1.25, 0.5, 0.5, 0, 60, 1], uppers=[0.2, 2.5, 1, 1, 0.2, 100, 2] -> -0.2 0 0 0 -0.2 20 0 0 1.25 0.5 0.5 0 60 1 0.2 2.5 1 1 0.2 100 2
- Univariate normal distribution, 
\tN(mean_brightness, variance_brightness);  N(mean_contrast, variance_contrast);  N(mean_horizontal_flip, variance_horizontal_flip);  N(mean_vertical_flip, variance_vertical_flip);  N(mean_hue, variance_hue);  N(mean_jpeg_quality, variance_jpeg_quality);  N(mean_saturation, variance_saturation)
\tstandard distribution parameters, N(0, 0.15);  N(1.25, 1);  N(0.5, 0.1);  N(0.5, 0.1);  N(0, 0.15);  N(60, 25);  N(1.25, 1.125)
\t\tLinearized version: Given means=[0, 1.25, 0.5, 0.5, 0, 60, 1.25], variances=[0.15, 1, 0.1, 0.1, 0.15, 25, 1.125] -> 0 1.25 0.5 0.5 0 60 1.25 0.15 1 0.1 0.1 0.15 25 1.125
- Multivariate normal distribution,
\tN(mean_vector, variance_covariance_matrix), with mean_vector = [mean_i] and variance_covariance_matrix = [var_ii or covar_ij], with i, j = {brightness, contrast, horizontally flip, vertically flip, hue, jpeg quality, saturation}
\tstandard distribution parameters, N(
                                        [0, 1.25, 0.5, 0.5, 0, 60, 1.25],
                                        [
                                            [0.15, 0, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 0],
                                            [0, 0, 0.1, 0, 0, 0, 0],
                                            [0, 0, 0, 0.1, 0, 0, 0],
                                            [0, 0, 0, 0, 0.15, 0, 0],
                                            [0, 0, 0, 0, 0, 25, 0],
                                            [0, 0, 0, 0, 0, 0, 1.125]
                                        ] 
                                    )
\t\tLinearized version: Given mean_vector=[0, 1.25, 0.5, 0.5, 0, 60, 1.25], variance_covariance_matrix=[[0.15, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0.1, 0, 0, 0, 0], [0, 0, 0, 0.1, 0, 0, 0], [0, 0, 0, 0, 0.15, 0, 0], [0, 0, 0, 0, 0, 25, 0], [0, 0, 0, 0, 0, 0, 1.125]] -> 0.15, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 1.125
"""
standard_ranges__low = """
Adaptive domain randomization, standard distribution parameters low ranges.

The meaning of the parameter depends on the distribution(s) considered. If univariate distributions are used, the order reflects the corresponding image parameter to be randomized:
- Univariate uniform distribution, 
\tU(lower_brightness, upper_brightness);  U(lower_contrast, upper_contrast);  U(lower_horizontal_flip, upper_horizontal_flip);  U(lower_vertical_flip, upper_vertical_flip);  U(lower_hue, upper_hue);  U(lower_jpeg_quality, upper_jpeg_quality);  U(lower_saturation, upper_saturation)
\tstandard distribution parameters low ranges, U(a, b), lower range for a = -1, lower range for b = -1;  U(a, b), lower range for a = float('-inf'), lower range for b = float('-inf');  U(a, b), lower range for a = float('-inf'), lower range for b = float('-inf');  U(a, b), lower range for a = float('-inf'), lower range for b = float('-inf');  U(a, b), lower range for a = -1, lower range for b = -1;  U(a, b), lower range for a = 0, lower range for b = 0;  U(a, b), lower range for a = 0, lower range for b = 0
\t\tLinearized version: Given lowers=[-1, float(-inf), float(-inf), float(-inf), -1, 0, 0], uppers= [-1, float(-inf), float(-inf), float(-inf), -1, 0, 0] -> -1 float(-inf) float(-inf) float(-inf) -1 0 0 -1 float(-inf) float(-inf) float(-inf) -1 0 0
- Univariate triangular distribution, 
\tTr(lower_brightness, mode_brightness, upper_brightness);  Tr(lower_contrast, mode_contrast, upper_contrast);  Tr(lower_horizontal_flip, mode_horizontal_flip, upper_horizontal_flip);  Tr(lower_vertical_flip, mode_vertical_flip, upper_vertical_flip);  Tr(lower_hue, mode_hue, upper_hue);  Tr(lower_jpeg_quality, mode_jpeg_quality, upper_jpeg_quality);  Tr(lower_saturation, mode_saturation, upper_saturation)
\tstandard distribution parameters low ranges, Tr(a, m, b), lower range for a = -1, lower range for m = -1, lower range for b = -1;  Tr(a, m, b), lower range for a = float('-inf'), lower range for m = float('-inf'), lower range for b = float('-inf');  Tr(a, m, b), lower range for a = float('-inf'), lower range for m = float('-inf'), lower range for b = float('-inf');  Tr(a, m, b), lower range for a = float('-inf'), lower range for m = float('-inf'), lower range for b = float('-inf');  Tr(a, m, b), lower range for a = -1, lower range for m = -1, lower range for b = -1;  Tr(a, m, b), lower range for a = 0, lower range for m = 0, lower range for b = 0;  Tr(a, m, b), lower range for a = 0, lower range for m = 0, lower range for b = 0
\t\tLinearized version: Given lowers=[-1, float(-inf), float(-inf), float(-inf), -1, 0, 0], modes= [-1, float(-inf), float(-inf), float(-inf), -1, 0, 0], uppers[-1, float(-inf), float(-inf), float(-inf), -1, 0, 0] -> -1 float(-inf) float(-inf) float(-inf) -1 0 0 -1 float(-inf) float(-inf) float(-inf) -1 0 0 -1 float(-inf) float(-inf) float(-inf) -1 0 0
- Univariate normal distribution, 
\tN(mean_brightness, variance_brightness);  N(mean_contrast, variance_contrast);  N(mean_horizontal_flip, variance_horizontal_flip);  N(mean_vertical_flip, variance_vertical_flip);  N(mean_hue, variance_hue);  N(mean_jpeg_quality, variance_jpeg_quality);  N(mean_saturation, variance_saturation)
\tstandard distribution parameters low ranges, N(mu, sigma), lower range for mu = -1, lower range for sigma = 0;  N(mu, sigma), lower range for mu = float('-inf'), lower range for sigma = 0;  N(mu, sigma), lower range for mu = float('-inf'), lower range for sigma = 0;  N(mu, sigma), lower range for mu = float('-inf'), lower range for sigma = 0;  N(mu, sigma), lower range for mu = -1, lower range for sigma = 0;  N(mu, sigma), lower range for mu = 0, lower range for sigma = 0;  N(mu, sigma), lower range for mu = 0, lower range for sigma = 0
\t\tLinearized version: Given means=[-1, float('-inf'), float('-inf'), float('-inf'), -1, 0, 0], variances=[0, 0, 0, 0, 0, 0, 0] -> -1 float('-inf') float('-inf') float('-inf') -1 0 0 0 0 0 0 0 0 0
- Multivariate normal distribution,
\tN(mean_vector, variance_covariance_matrix), with mean_vector = [mean_i] and variance_covariance_matrix = [var_ii or covar_ij], with i, j = {brightness, contrast, horizontally flip, vertically flip, hue, jpeg quality, saturation}
\tstandard distribution parameters low ranges, N(
                                        [mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7],
                                        [
                                            [sigma_11, sigma_12, sigma_13, sigma_14, sigma_15, sigma_16, sigma_17],
                                            [sigma_21, sigma_22, sigma_23, sigma_24, sigma_25, sigma_26, sigma_27],
                                            [sigma_31, sigma_32, sigma_33, sigma_34, sigma_35, sigma_36, sigma_37],
                                            [sigma_41, sigma_42, sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                                            [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55, sigma_56, sigma_57],
                                            [sigma_61, sigma_62, sigma_63, sigma_64, sigma_65, sigma_66, sigma_67],
                                            [sigma_71, sigma_72, sigma_73, sigma_74, sigma_75, sigma_76, sigma_77]
                                        ]
                                                ), lower range for mu_1 = -1, lower range for mu_2 = float('-inf'), lower range for mu_3 = float('-inf'), lower range for mu_4 = float('-inf'), lower range for mu_5 = -1, lower range for mu_6 = 0, lower range for mu_7 = 0, lower range for sigma_ij = 0 if i = j, lower range for sigma_ij = -100 if i != j
\t\tLinearized version: Given mean_vector=[-1, float('-inf'), float('-inf'), float('-inf'), -1, 0, 0], variance_covariance_matrix=[[0], [-100, 0], [-100, -100, 0], [-100, -100, -100, 0], [-100, -100, -100, -100, 0], [-100, -100, -100, -100, -100, 0], [-100, -100, -100, -100, -100, -100, 0]] -> -1 float('-inf') float('-inf') float('-inf') -1 0 0 0 -100 0 -100 -100 0 -100 -100 -100 0 -100 -100 -100 -100 0 -100 -100 -100 -100 -100 0 -100 -100 -100 -100 -100 -100 0 -- triangular matrix required -- 
"""
standard_ranges__up = """
Adaptive domain randomization, standard distribution parameters up ranges.

The meaning of the parameter depends on the distribution(s) considered. If univariate distributions are used, the order reflects the corresponding image parameter to be randomized:
- Univariate uniform distribution, 
\tU(lower_brightness, upper_brightness);  U(lower_contrast, upper_contrast);  U(lower_horizontal_flip, upper_horizontal_flip);  U(lower_vertical_flip, upper_vertical_flip);  U(lower_hue, upper_hue);  U(lower_jpeg_quality, upper_jpeg_quality);  U(lower_saturation, upper_saturation)
\tstandard distribution parameters up ranges, U(a, b), upper range for a = 1, upper range for b = 1;  U(a, b), upper range for a = float('inf'), upper range for b = float('inf');  U(a, b), upper range for a = float('inf'), upper range for b = float('inf');  U(a, b), upper range for a = float('inf'), upper range for b = float('inf');  U(a, b), upper range for a = 1, upper range for b = 1;  U(a, b), upper range for a = 100, upper range for b = 100;  U(a, b), upper range for a = float('inf'), upper range for b = float('inf')
\t\tLinearized version: Given lowers=[1, float(inf), float(inf), float(inf), 1, 100, float(inf)], uppers= [1, float(inf), float(inf), float(inf), 1, 100, float(inf)] -> 1 float(inf) float(inf) float(inf) 1 100 float(inf) 1 float(inf) float(inf) float(inf) 1 100 float(inf)
- Univariate triangular distribution, 
\tTr(lower_brightness, mode_brightness, upper_brightness);  Tr(lower_contrast, mode_contrast, upper_contrast);  Tr(lower_horizontal_flip, mode_horizontal_flip, upper_horizontal_flip);  Tr(lower_vertical_flip, mode_vertical_flip, upper_vertical_flip);  Tr(lower_hue, mode_hue, upper_hue);  Tr(lower_jpeg_quality, mode_jpeg_quality, upper_jpeg_quality);  Tr(lower_saturation, mode_saturation, upper_saturation)
\tstandard distribution parameters up ranges, Tr(a, m, b), upper range for a = 1, upper range for m = 1, upper range for b = 1;  Tr(a, m, b), upper range for a =  float('inf'), upper range for m =  float('inf'), upper range for b =  float('inf');  Tr(a, m, b), upper range for a =  float('inf'), upper range for m =  float('inf'), upper range for b =  float('inf');  Tr(a, m, b), upper range for a =  float('inf'), upper range for m =  float('inf'), upper range for b =  float('inf');  Tr(a, m, b), upper range for a = 1, upper range for m = 1, upper range for b = 1;  Tr(a, m, b), upper range for a = 100, upper range for m = 100, upper range for b = 100;  Tr(a, m, b), upper range for a =  float('inf'), upper range for m =  float('inf'), upper range for b =  float('inf')
\t\tLinearized version: Given lowers=[1, float(inf), float(inf), float(inf), 1, 100, float(inf)], modes= [1, float(inf), float(inf), float(inf), 1, 100, float(inf)], uppers[1, float(inf), float(inf), float(inf), 1, 100, float(inf)] -> 1 float(inf) float(inf) float(inf) 1 100 float(inf) 1 float(inf) float(inf) float(inf) 1 100 float(inf) 1 float(inf) float(inf) float(inf) 1 100 float(inf)
- Univariate normal distribution, 
\tN(mean_brightness, variance_brightness);  N(mean_contrast, variance_contrast);  N(mean_horizontal_flip, variance_horizontal_flip);  N(mean_vertical_flip, variance_vertical_flip);  N(mean_hue, variance_hue);  N(mean_jpeg_quality, variance_jpeg_quality);  N(mean_saturation, variance_saturation)
\tstandard distribution parameters up ranges, N(mu, sigma), upper range for mu = 1, upper range for sigma = 0.4;  N(mu, sigma), upper range for mu = float('inf'), upper range for sigma = 10;  N(mu, sigma), upper range for mu = float('inf'), upper range for sigma = 10;  N(mu, sigma), upper range for mu = float('inf'), upper range for sigma = 10;  N(mu, sigma), upper range for mu = 1, upper range for sigma = 0.4;  N(mu, sigma), upper range for mu = 100, upper range for sigma = 25;  N(mu, sigma), upper range for mu = float('inf'), upper range for sigma = 10
\t\tLinearized version: Given means=[1, float('inf'), float('inf'), float('inf'), 1, 100, float('inf')], variances=[0.4, 10, 10, 10, 0.4, 25, 10] -> 1 float('inf') float('inf') float('inf') 1 100 float('inf') 0.4 10 10 10 0.4 25 10
- Multivariate normal distribution,
\tN(mean_vector, variance_covariance_matrix), with mean_vector = [mean_i] and variance_covariance_matrix = [var_ii or covar_ij], with i, j = {brightness, contrast, horizontally flip, vertically flip, hue, jpeg quality, saturation}
\tstandard distribution parameters up ranges, N(
                                        [mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7],
                                        [
                                            [sigma_11, sigma_12, sigma_13, sigma_14, sigma_15, sigma_16, sigma_17],
                                            [sigma_21, sigma_22, sigma_23, sigma_24, sigma_25, sigma_26, sigma_27],
                                            [sigma_31, sigma_32, sigma_33, sigma_34, sigma_35, sigma_36, sigma_37],
                                            [sigma_41, sigma_42, sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                                            [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55, sigma_56, sigma_57],
                                            [sigma_61, sigma_62, sigma_63, sigma_64, sigma_65, sigma_66, sigma_67],
                                            [sigma_71, sigma_72, sigma_73, sigma_74, sigma_75, sigma_76, sigma_77]
                                        ]
                                                ), upper range for mu_1 = 1, upper range for mu_2 = float('inf'), upper range for mu_3 = float('inf'), upper range for mu_4 = float('inf'), upper range for mu_5 = 1, upper range for mu_6 = 100, upper range for mu_7 = float('inf'), upper range for sigma_ij = 0.4 if i = j = {1, 5}, upper range for sigma_ij = 10 if i = j = {2, 3, 4, 7}, upper range for sigma_ij = 25 if i = j = 6, upper range for sigma_ij = 100 if i != j
\t\tLinearized version: Given mean_vector=[1, float('inf'), float('inf'), float('inf'), 1, 100, float('inf')], variance_covariance_matrix=[[0.4], [100, 10], [100, 100, 10], [100, 100, 100, 10], [100, 100, 100, 100, 0.4], [100, 100, 100, 100, 100, 25], [100, 100, 100, 100, 100, 100, 10]] -> 1 float('inf') float('inf') float('inf') 1 100 float('inf') 0.4 100 10 100 100 10 100 100 100 10 100 100 100 100 0.4 100 100 100 100 100 25 100 100 100 100 100 100 10 -- triangular matrix required -- 
"""
standard_initials = """
Adaptive domain randomization, standard distribution parameters initial values.

The meaning of the parameter depends on the distribution(s) considered. If univariate distributions are used, the order reflects the corresponding image parameter to be randomized:
- Univariate uniform distribution, 
\tU(lower_brightness, upper_brightness);  U(lower_contrast, upper_contrast);  U(lower_horizontal_flip, upper_horizontal_flip);  U(lower_vertical_flip, upper_vertical_flip);  U(lower_hue, upper_hue);  U(lower_jpeg_quality, upper_jpeg_quality);  U(lower_saturation, upper_saturation)
\tstandard distribution parameters initial values, U(a, b), initial value for a = 0, initial value for b = 0;  U(a, b), initial value for a = 0, initial value for b = 0;  U(a, b), initial value for a = 0, initial value for b = 0;  U(a, b), initial value for a = 0, initial value for b = 0;  U(a, b), initial value for a = 0, initial value for b = 0;  U(a, b), initial value for a = 50, initial value for b = 50;  U(a, b), initial value for a = 1.25, initial value for b = 1.25
\t\tLinearized version: Given lowers=[0, 0, 0, 0, 0, 50, 1.25], uppers= [0, 0, 0, 0, 0, 50, 1.25] -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25
- Univariate triangular distribution, 
\tTr(lower_brightness, mode_brightness, upper_brightness);  Tr(lower_contrast, mode_contrast, upper_contrast);  Tr(lower_horizontal_flip, mode_horizontal_flip, upper_horizontal_flip);  Tr(lower_vertical_flip, mode_vertical_flip, upper_vertical_flip);  Tr(lower_hue, mode_hue, upper_hue);  Tr(lower_jpeg_quality, mode_jpeg_quality, upper_jpeg_quality);  Tr(lower_saturation, mode_saturation, upper_saturation)
\tstandard distribution parameters initial values, Tr(a, m, b), initial value for a = 0, initial value for m = 0, initial value for b = 0;  Tr(a, m, b), initial value for a = 0, initial value for m = 0, initial value for b = 0;  Tr(a, m, b), initial value for a = 0, initial value for m = 0, initial value for b = 0;  Tr(a, m, b), initial value for a = 0, initial value for m = 0, initial value for b = 0;  Tr(a, m, b), initial value for a = 0, initial value for m = 0, initial value for b = 0;  Tr(a, m, b), initial value for a = 50, initial value for m = 50, initial value for b = 50;  Tr(a, m, b), initial value for a = 1.25, initial value for m = 1.25, initial value for b = 1.25
\t\tLinearized version: Given lowers=[0, 0, 0, 0, 0, 50, 1.25], modes= [0, 0, 0, 0, 0, 50, 1.25], uppers=[0, 0, 0, 0, 0, 50, 1.25] -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25
- Univariate normal distribution, 
\tN(mean_brightness, variance_brightness);  N(mean_contrast, variance_contrast);  N(mean_horizontal_flip, variance_horizontal_flip);  N(mean_vertical_flip, variance_vertical_flip);  N(mean_hue, variance_hue);  N(mean_jpeg_quality, variance_jpeg_quality);  N(mean_saturation, variance_saturation)
\tstandard distribution parameters initial values, N(mu, sigma), initial value for mu = 0, initial value for sigma = 0.2;  N(mu, sigma), initial value for mu = 0, initial value for sigma = 5;  N(mu, sigma), initial value for mu = 0, initial value for sigma = 5;  N(mu, sigma), initial value for mu = 0, initial value for sigma = 5;  N(mu, sigma), initial value for mu = 0, initial value for sigma = 0.2;  N(mu, sigma), initial value for mu = 50, initial value for sigma = 12.5;  N(mu, sigma), initial value for mu = 1.25, initial value for sigma = 5
\t\tLinearized version: Given means=[0, 0, 0, 0, 0, 50, 1.25], variances=[0.2, 5, 5, 5, 0.2, 12.5, 5] -> 0 0 0 0 0 50 1.25 0.2 5 5 5 0.2 12.5 5
- Multivariate normal distribution,
\tN(mean_vector, variance_covariance_matrix), with mean_vector = [mean_i] and variance_covariance_matrix = [var_ii or covar_ij], with i, j = {brightness, contrast, horizontally flip, vertically flip, hue, jpeg quality, saturation}
\tstandard distribution parameters initial values, N(
                                        [mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7],
                                        [
                                            [sigma_11, sigma_12, sigma_13, sigma_14, sigma_15, sigma_16, sigma_17],
                                            [sigma_21, sigma_22, sigma_23, sigma_24, sigma_25, sigma_26, sigma_27],
                                            [sigma_31, sigma_32, sigma_33, sigma_34, sigma_35, sigma_36, sigma_37],
                                            [sigma_41, sigma_42, sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                                            [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55, sigma_56, sigma_57],
                                            [sigma_61, sigma_62, sigma_63, sigma_64, sigma_65, sigma_66, sigma_67],
                                            [sigma_71, sigma_72, sigma_73, sigma_74, sigma_75, sigma_76, sigma_77]
                                        ]
                                                ), initial value for mu_1 = 0, initial value for mu_2 = 0, initial value for mu_3 = 0, initial value for mu_4 = 0, initial value for mu_5 = 0, initial value for mu_6 = 1.25, initial value for mu_7 = 50, initial value for sigma_ij = 0.2 if i = j = {1,5}, initial value for sigma_ij = 5 if i = j = {2,3,4,7}, initial value for sigma_ij = 12.5 if i = j = 6, initial value for sigma_ij = 0 if i != j
\t\tLinearized version: Given mean_vector=[0, 0, 0, 0, 0, 1.25, 50], variance_covariance_matrix=[[0.2], [0, 5], [0, 0, 5], [0, 0, 0, 5], [0, 0, 0, 0, 0.2], [0, 0, 0, 0, 0, 12.5], [0, 0, 0, 0, 0, 0, 5]] -> 0 0 0 0 0 1.25 500.2 0 5 0 0 5 0 0 0 5 0 0 0 0 0.2 0 0 0 0 0 12.5 0 0 0 0 0 0 5  -- triangular matrix required -- 
"""
parser.add_argument('--mean', type=str, nargs='+', help="Mean values for normalization. They can be multiple floats separated by a spaces (list representing the values, whose cardinality depends on the image format) or 'auto' (to apply standard values) or None (no normalization)\n.Standard values." + standard_norm + "\n(default: 'auto')", default=["auto"])
parser.add_argument('--std', type=str, nargs='+', help="Standard deviation values for normalization. They can be multiple floats separated by a spaces (list representing the values, whose cardinality depends on the image format) or 'auto' (to apply standard values) or None (no standard deviation, i.e.: sd = 0)\n.Standard values." + standard_norm + "\n(default: None)", default=[None])
parser.add_argument('--resize', action='store_true', help="Resize option to be applied to images of the dataset.\n(default: True)", default=True)
parser.add_argument('--height', type=str, help="Height value for resize. It can be a float or 'auto' (to apply standard values) or None (no resize)\n.Standard values." + standard_res + "\n(default: 'auto')", default="auto")
parser.add_argument('--width', type=str, help="Width value for resize. It can be a float or 'auto' (to apply standard values) or None (no resize)\n.Standard values." + standard_res + "\n(default: 'auto')", default="auto")
parser.add_argument('--one_hot_encoding', action='store_true', help="One-hot encoding option to be applied to labels\n(default: 'True')", default=True)
parser.add_argument('--name', type=str, help="Name of the dataset\n(default: 'Electronic components dataset')", default="Electronic components dataset")
parser.add_argument('--format', type=str, help="Image format (e.g.: RGB, Grayscale, RGBA). If not present, automatically deduced from images. Images of homegeneus formats are suggested to improve algorithm performances.\n(default: None)", default=None)
parser.add_argument('--buffer_size', type=int, help='Buffer size\n(default:500)', default=500)
parser.add_argument('--batch_size', type=str, help="Batch size. Also possibile to use 'no batches'\n(default:16)", default=16)
parser.add_argument('--model_path', type=str, help="Path of the directory in which the trained model will be saved.\n(default: out\model)", default="out\\model")
parser.add_argument('--domain_randomization', action='store_true', help="Enable image parameters domain randomization." + image_params + "\n(default: True)", default=True)
parser.add_argument('--dom_rand__mode', type=str, help="Domain randomization mode." + dr__mode + "\n(default: multivariate normal)", default="multivariate normal")
parser.add_argument('--dom_rand__seed', type=int, help="Domain randomization random seed.\n(default: None)", default=None)
parser.add_argument('--dom_rand__factors', type=float, nargs='+', help="Domain randomization factor parameters. Multiple floats separated by a spaces (list representing the values, whose cardinality depends on the number of parameters). If None, standard factors are used." + standard_facts + "\n(default: None)", default=[None])
parser.add_argument('--dom_rand__dist_pars', type=float, nargs='+', help="Domain randomization distribution parameters. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distributions parameters are used." + standard_pars + "\n(default: None)", default=[None])
parser.add_argument('--dom_rand__dist_pars___uniform', type=float, nargs='+', help="Domain randomization uniform distribution parameters. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distributions parameters are used." + standard_pars + "\n(default: None)", default=[None])
parser.add_argument('--dom_rand__dist_pars___triangular', type=float, nargs='+', help="Domain randomization triangular distribution parameters. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distributions parameters are used." + standard_pars + "\n(default: None)", default=[None])
parser.add_argument('--dom_rand__dist_pars___univariatenormal', type=float, nargs='+', help="Domain randomization univatiate normal distribution parameters. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distributions parameters are used." + standard_pars + "\n(default: None)", default=[None])
parser.add_argument('--dom_rand__dist_pars___multivariatenormal', type=float, nargs='+', help="Domain randomization multivatiate normal distribution parameters. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distributions parameters are used." + standard_pars + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand', action='store_true', help="Optimize domain randomization distribution parameters.\n(default: False)", default=False)
parser.add_argument('--adaptive_dom_rand__dist_ranges____low', type=float, nargs='+', help="Adaptive domain randomization distribution parameters low ranges. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_ranges__low + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_ranges____up', type=float, nargs='+', help="Adaptive domain randomization distribution parameters up ranges. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_ranges__up + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_ranges___uniform____low', type=float, nargs='+', help="Adaptive domain randomization uniform distribution parameters low ranges. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_ranges__low + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_ranges___uniform____up', type=float, nargs='+', help="Adaptive domain randomization uniform distribution parameters up ranges. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_ranges__up + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_ranges___triangular____low', type=float, nargs='+', help="Adaptive domain randomization triangular distribution parameters low ranges. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_ranges__low + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_ranges___triangular____up', type=float, nargs='+', help="Adaptive domain randomization triangular distribution parameters up ranges. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_ranges__up + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_ranges___univariatenormal____low', type=float, nargs='+', help="Adaptive domain randomization univatiate normal distribution parameters low ranges. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_ranges__low + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_ranges___univariatenormal____up', type=float, nargs='+', help="Adaptive domain randomization univatiate normal distribution parameters up ranges. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_ranges__up + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_ranges___multivariatenormal____low', type=float, nargs='+', help="Adaptive domain randomization multivatiate normal distribution parameters low ranges. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_ranges__low + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_ranges___multivariatenormal____up', type=float, nargs='+', help="Adaptive domain randomization multivatiate normal distribution parameters up ranges. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_ranges__up + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_initials', type=float, nargs='+', help="Adaptive domain randomization distribution parameters initial values. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_initials + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_initials___uniform', type=float, nargs='+', help="Adaptive domain randomization uniform distribution parameters initial values. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_initials + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_initials___triangular', type=float, nargs='+', help="Adaptive domain randomization triangular distribution parameters initial values. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_initials + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_initials___univariatenormal', type=float, nargs='+', help="Adaptive domain randomization univatiate normal distribution parameters initial values. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_initials + "\n(default: None)", default=[None])
parser.add_argument('--adaptive_dom_rand__dist_initials___multivariatenormal', type=float, nargs='+', help="Adaptive domain randomization multivatiate normal distribution parameters initial values. Use the linearized version (i.e. Parameters list, whose cardinality depends on the number of parameters: single parameters separated by 1 single space. A single parameter can be a scalar, a vector or a a matrix. Inside a vector, elements are separated by 1 single space. Inside a matrix, rows are representd as a (transpose) vector and inserted separated by 1 single space). If None, standard distribution parameters ranges are used." + standard_initials + "\n(default: None)", default=[None])
parser.add_argument('--neural_network', type=str, help="Neural Network to be used. Possible choices: [ResNet 1, ResNet 2.0, ResNet 2.1, ResNet 2.0.1, ResNet 2.1.1].\n" + nn_desc + "(default: ResNet 2.0)", default="ResNet 2.0")
gradient_based__optimizers = {
    'SGD': keras.optimizers.SGD,
    'RMSprop': keras.optimizers.RMSprop,
    'Adagrad': keras.optimizers.Adagrad,
    'Adadelta': keras.optimizers.Adadelta,
    'Adam': keras.optimizers.Adam,
    'Adamax': keras.optimizers.Adamax,
    'Nadam': keras.optimizers.Nadam,
    'FTRL': keras.optimizers.FTRL,
    'ProximalGradientDescent': keras.optimizers.ProximalGradientDescent,
    'ProximalAdagrad': keras.optimizers.ProximalAdagrad,
    'Schedules': keras.optimizers.Schedules
}
gradient_free__optimizers = {
    'RandomSearch': RandomSearch,
    'QORandomSearch': QORandomSearch,
    'ORandomSearch': ORandomSearch,
    'RandomSearchPlusMiddlePoint': RandomSearchPlusMiddlePoint,
    'MetaRecentering': MetaRecentering,
    'MetaTuneRecentering': MetaTuneRecentering,
    'HullAvgMetaTuneRecentering': HullAvgMetaTuneRecentering,
    'HullAvgMetaRecentering': HullAvgMetaRecentering,
    'AvgMetaRecenteringNoHull': AvgMetaRecenteringNoHull,
    'HaltonSearch': HaltonSearch,
    'HaltonSearchPlusMiddlePoint': HaltonSearchPlusMiddlePoint,
    'LargeHaltonSearch': LargeHaltonSearch,
    'ScrHaltonSearch': ScrHaltonSearch,
    'ScrHaltonSearchPlusMiddlePoint': ScrHaltonSearchPlusMiddlePoint,
    'HammersleySearch': HammersleySearch,
    'HammersleySearchPlusMiddlePoint': HammersleySearchPlusMiddlePoint,
    'ScrHammersleySearchPlusMiddlePoint': ScrHammersleySearchPlusMiddlePoint,
    'ScrHammersleySearch': ScrHammersleySearch,
    'QOScrHammersleySearch': QOScrHammersleySearch,
    'OScrHammersleySearch': OScrHammersleySearch,
    'CauchyScrHammersleySearch': CauchyScrHammersleySearch,
    'LHSSearch': LHSSearch,
    'CauchyLHSSearch': CauchyLHSSearch,
    'DE': DE,
    'TwoPointsDE': TwoPointsDE,
    'VoronoiDE': VoronoiDE,
    'RotatedTwoPointsDE': RotatedTwoPointsDE,
    'LhsDE': LhsDE,
    'QrDE': QrDE,
    'QODE': QODE,
    'SPQODE': SPQODE,
    'QOTPDE': QOTPDE,
    'LQOTPDE': LQOTPDE,
    'LQODE': LQODE,
    'SODE': SODE,
    'NoisyDE': NoisyDE,
    'AlmostRotationInvariantDE': AlmostRotationInvariantDE,
    'RotationInvariantDE': RotationInvariantDE,
    'DiscreteDE': DiscreteDE,
    'RecES': RecES,
    'RecMixES': RecMixES,
    'RecMutDE': RecMutDE,
    'ES': ES,
    'MixES': MixES,
    'MutDE': MutDE,
    'NonNSGAIIES': NonNSGAIIES,
    'AX': AX,
    'BOBYQA': BOBYQA,
    'NelderMead': NelderMead,
    'CmaFmin2': CmaFmin2,
    'Powell': Powell,
    'RPowell': RPowell,
    'BFGS': BFGS,
    'RBFGS': RBFGS,
    'LBFGSB': LBFGSB,
    'Cobyla': Cobyla,
    'RCobyla': RCobyla,
    'SQP': SQP,
    'RSQP': RSQP,
    'NLOPT_LN_SBPLX': NLOPT_LN_SBPLX,
    'NLOPT_LN_PRAXIS': NLOPT_LN_PRAXIS,
    'NLOPT_GN_DIRECT': NLOPT_GN_DIRECT,
    'NLOPT_GN_DIRECT_L': NLOPT_GN_DIRECT_L,
    'NLOPT_GN_CRS2_LM': NLOPT_GN_CRS2_LM,
    'NLOPT_GN_AGS': NLOPT_GN_AGS,
    'NLOPT_GN_ISRES': NLOPT_GN_ISRES,
    'NLOPT_GN_ESCH': NLOPT_GN_ESCH,
    'NLOPT_LN_COBYLA': NLOPT_LN_COBYLA,
    'NLOPT_LN_BOBYQA': NLOPT_LN_BOBYQA,
    'NLOPT_LN_NEWUOA_BOUND': NLOPT_LN_NEWUOA_BOUND,
    'NLOPT_LN_NELDERMEAD': NLOPT_LN_NELDERMEAD,
    'SMAC3': SMAC3,
    'PymooCMAES': PymooCMAES,
    'PymooBIPOP': PymooBIPOP,
    'PymooNSGA2': PymooNSGA2,
    'pysot': pysot,
    'DSbase': DSbase,
    'DS3p': DS3p,
    'DSsubspace': DSsubspace,
    'DSproba': DSproba,
    'DSproba2': DSproba2,
    'DSproba3': DSproba3,
    'DSproba4': DSproba4,
    'DSproba5': DSproba5,
    'DSproba6': DSproba6,
    'DSproba7': DSproba7,
    'DSproba8': DSproba8,
    'DSproba9': DSproba9,
    'OnePlusOne': OnePlusOne,
    'OnePlusLambda': OnePlusLambda,
    'NoisyOnePlusOne': NoisyOnePlusOne,
    'DiscreteOnePlusOne': DiscreteOnePlusOne,
    'SADiscreteLenglerOnePlusOneExp09': SADiscreteLenglerOnePlusOneExp09,
    'SADiscreteLenglerOnePlusOneExp099': SADiscreteLenglerOnePlusOneExp099,
    'SADiscreteLenglerOnePlusOneExp09Auto': SADiscreteLenglerOnePlusOneExp09Auto,
    'SADiscreteLenglerOnePlusOneLinAuto': SADiscreteLenglerOnePlusOneLinAuto,
    'SADiscreteLenglerOnePlusOneLin1': SADiscreteLenglerOnePlusOneLin1,
    'SADiscreteLenglerOnePlusOneLin100': SADiscreteLenglerOnePlusOneLin100,
    'SADiscreteOnePlusOneExp099': SADiscreteOnePlusOneExp099,
    'SADiscreteOnePlusOneLin100': SADiscreteOnePlusOneLin100,
    'SADiscreteOnePlusOneExp09': SADiscreteOnePlusOneExp09,
    'DiscreteOnePlusOneT': DiscreteOnePlusOneT,
    'PortfolioDiscreteOnePlusOne': PortfolioDiscreteOnePlusOne,
    'PortfolioDiscreteOnePlusOneT': PortfolioDiscreteOnePlusOneT,
    'DiscreteLenglerOnePlusOne': DiscreteLenglerOnePlusOne,
    'DiscreteLengler2OnePlusOne': DiscreteLengler2OnePlusOne,
    'DiscreteLengler3OnePlusOne': DiscreteLengler3OnePlusOne,
    'DiscreteLenglerHalfOnePlusOne': DiscreteLenglerHalfOnePlusOne,
    'DiscreteLenglerFourthOnePlusOne': DiscreteLenglerFourthOnePlusOne,
    'DiscreteLenglerOnePlusOneT': DiscreteLenglerOnePlusOneT,
    'AdaptiveDiscreteOnePlusOne': AdaptiveDiscreteOnePlusOne,
    'LognormalDiscreteOnePlusOne': LognormalDiscreteOnePlusOne,
    'AnisotropicAdaptiveDiscreteOnePlusOne': AnisotropicAdaptiveDiscreteOnePlusOne,
    'DiscreteBSOOnePlusOne': DiscreteBSOOnePlusOne,
    'DiscreteDoerrOnePlusOne': DiscreteDoerrOnePlusOne,
    'CauchyOnePlusOne': CauchyOnePlusOne,
    'OptimisticNoisyOnePlusOne': OptimisticNoisyOnePlusOne,
    'OptimisticDiscreteOnePlusOne': OptimisticDiscreteOnePlusOne,
    'NoisyDiscreteOnePlusOne': NoisyDiscreteOnePlusOne,
    'DoubleFastGADiscreteOnePlusOne': DoubleFastGADiscreteOnePlusOne,
    'RLSOnePlusOne': RLSOnePlusOne,
    'SparseDoubleFastGADiscreteOnePlusOne': SparseDoubleFastGADiscreteOnePlusOne,
    'RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne': RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne,
    'RecombiningPortfolioDiscreteOnePlusOne': RecombiningPortfolioDiscreteOnePlusOne,
    'ChoiceBase': ng.optimization.optimizerlib.ChoiceBase,
    'OldCMA': OldCMA,
    'LargeCMA': LargeCMA,
    'LargeDiagCMA': LargeDiagCMA,
    'TinyCMA': TinyCMA,
    'CMAbounded': CMAbounded,
    'CMAsmall': CMAsmall,
    'CMAstd': CMAstd,
    'CMApara': CMApara,
    'CMAtuning': CMAtuning,
    'MetaCMA': ng.optimization.optimizerlib.MetaCMA,
    'DiagonalCMA': DiagonalCMA,
    'SDiagonalCMA': SDiagonalCMA,
    'FCMA': FCMA,
    'CMA': ng.optimization.optimizerlib.CMA,
    'EDA': ng.optimization.optimizerlib.EDA,
    'TBPSA': TBPSA,
    'NaiveTBPSA': NaiveTBPSA,
    'NoisyBandit': ng.optimization.optimizerlib.NoisyBandit,
    'RealSpacePSO': RealSpacePSO,
    'PSO': PSO,
    'QOPSO': QOPSO,
    'QORealSpacePSO': QORealSpacePSO,
    'SQOPSO': SQOPSO,
    'SOPSO': SOPSO,
    'SQORealSpacePSO': SQORealSpacePSO,
    'SPSA': ng.optimization.optimizerlib.SPSA,
    'RescaledCMA': RescaledCMA,
    'TinyLhsDE': TinyLhsDE,
    'TinyQODE': TinyQODE,
    'TinySQP': TinySQP,
    'MicroSQP': MicroSQP,
    'TinySPSA': TinySPSA,
    'MicroSPSA': MicroSPSA,
    'VastLengler': VastLengler,
    'VastDE': VastDE,
    'Portfolio': ng.optimization.optimizerlib.Portfolio,
    'ParaPortfolio': ParaPortfolio,
    'ASCMADEthird': ASCMADEthird,
    'MultiCMA': MultiCMA,
    'MultiDS': MultiDS,
    'TripleCMA': TripleCMA,
    'PolyCMA': PolyCMA,
    'MultiScaleCMA': MultiScaleCMA,
    'LPCMA': LPCMA,
    'VLPCMA': VLPCMA,
    'MetaModel': MetaModel,
    'NeuralMetaModel': NeuralMetaModel,
    'SVMMetaModel': SVMMetaModel,
    'RFMetaModel': RFMetaModel,
    'MetaModelOnePlusOne': MetaModelOnePlusOne,
    'MetaModelDSproba': MetaModelDSproba,
    'RFMetaModelOnePlusOne': RFMetaModelOnePlusOne,
    'MetaModelPSO': MetaModelPSO,
    'RFMetaModelPSO': RFMetaModelPSO,
    'SVMMetaModelPSO': SVMMetaModelPSO,
    'MetaModelDE': MetaModelDE,
    'MetaModelQODE': MetaModelQODE,
    'NeuralMetaModelDE': NeuralMetaModelDE,
    'SVMMetaModelDE': SVMMetaModelDE,
    'RFMetaModelDE': RFMetaModelDE,
    'MetaModelTwoPointsDE': MetaModelTwoPointsDE,
    'NeuralMetaModelTwoPointsDE': NeuralMetaModelTwoPointsDE,
    'SVMMetaModelTwoPointsDE': SVMMetaModelTwoPointsDE,
    'RFMetaModelTwoPointsDE': RFMetaModelTwoPointsDE,
    'MultiBFGSPlus': ng.optimization.optimizerlib.MultiBFGSPlus,
    'LogMultiBFGSPlus': ng.optimization.optimizerlib.LogMultiBFGSPlus,
    'SqrtMultiBFGSPlus': ng.optimization.optimizerlib.SqrtMultiBFGSPlus,
    'MultiCobylaPlus': ng.optimization.optimizerlib.MultiCobylaPlus,
    'MultiSQPPlus': ng.optimization.optimizerlib.MultiSQPPlus,
    'BFGSCMAPlus': ng.optimization.optimizerlib.BFGSCMAPlus,
    'LogBFGSCMAPlus': ng.optimization.optimizerlib.LogBFGSCMAPlus,
    'SqrtBFGSCMAPlus': ng.optimization.optimizerlib.SqrtBFGSCMAPlus,
    'SQPCMAPlus': ng.optimization.optimizerlib.SQPCMAPlus,
    'LogSQPCMAPlus': ng.optimization.optimizerlib.LogSQPCMAPlus,
    'SqrtSQPCMAPlus': ng.optimization.optimizerlib.SqrtSQPCMAPlus,
    'MultiBFGS': ng.optimization.optimizerlib.MultiBFGS,
    'LogMultiBFGS': ng.optimization.optimizerlib.LogMultiBFGS,
    'SqrtMultiBFGS': ng.optimization.optimizerlib.SqrtMultiBFGS,
    'MultiCobyla': ng.optimization.optimizerlib.MultiCobyla,
    'ForceMultiCobyla': ng.optimization.optimizerlib.ForceMultiCobyla,
    'MultiSQP': ng.optimization.optimizerlib.MultiSQP,
    'BFGSCMA': ng.optimization.optimizerlib.BFGSCMA,
    'LogBFGSCMA': ng.optimization.optimizerlib.LogBFGSCMA,
    'SqrtBFGSCMA': ng.optimization.optimizerlib.SqrtBFGSCMA,
    'SQPCMA': ng.optimization.optimizerlib.SQPCMA,
    'LogSQPCMA': ng.optimization.optimizerlib.LogSQPCMA,
    'SqrtSQPCMA': ng.optimization.optimizerlib.SqrtSQPCMA,
    'FSQPCMA': ng.optimization.optimizerlib.FSQPCMA,
    'F2SQPCMA': ng.optimization.optimizerlib.F2SQPCMA,
    'F3SQPCMA': ng.optimization.optimizerlib.F3SQPCMA,
    'MultiDiscrete': ng.optimization.optimizerlib.MultiDiscrete,
    'CMandAS2': ng.optimization.optimizerlib.CMandAS2,
    'CMandAS3': ng.optimization.optimizerlib.CMandAS3,
    'CM': ng.optimization.optimizerlib.CM,
    'BO': BO,
    'BOSplit': BOSplit,
    'PCABO': PCABO,
    'BayesOptimBO': BayesOptimBO,
    'GeneticDE': GeneticDE,
    'MemeticDE': MemeticDE,
    'QNDE': QNDE,
    'ChainDE': ChainDE,
    'OpoDE': OpoDE,
    'OpoTinyDE': OpoTinyDE,
    'Carola1': Carola1,
    'Carola2': Carola2,
    'DS2': DS2,
    'Carola4': Carola4,
    'DS4': DS4,
    'Carola5': Carola5,
    'DS5': DS5,
    'Carola6': Carola6,
    'DS6': DS6,
    'PCarola6': PCarola6,
    'pCarola6': pCarola6,
    'Carola7': Carola7,
    'Carola8': Carola8,
    'DS8': DS8,
    'Carola9': Carola9,
    'DS9': DS9,
    'Carola10': Carola10,
    'Carola3': ng.optimization.optimizerlib.Carola3,
    'BAR': BAR,
    'BAR2': BAR2,
    'BAR3': BAR3,
    'discretememetic': discretememetic,
    'ChainCMAPowell': ChainCMAPowell,
    'ChainDSPowell': ChainDSPowell,
    'ChainMetaModelSQP': ChainMetaModelSQP,
    'ChainMetaModelDSSQP': ChainMetaModelDSSQP,
    'ChainMetaModelPowell': ChainMetaModelPowell,
    'ChainDiagonalCMAPowell': ChainDiagonalCMAPowell,
    'ChainNaiveTBPSAPowell': ChainNaiveTBPSAPowell,
    'ChainNaiveTBPSACMAPowell': ChainNaiveTBPSACMAPowell,
    'BAR4': BAR4,
    'cGA': ng.optimization.optimizerlib.cGA,
    'NaiveIsoEMNA': NaiveIsoEMNA,
    'NGOptBase': ng.optimization.optimizerlib.NGOptBase,
    'NGOptDSBase': ng.optimization.optimizerlib.NGOptDSBase,
    'Shiwa': ng.optimization.optimizerlib.Shiwa,
    'NGO': ng.optimization.optimizerlib.NGO,
    'NGOpt4': ng.optimization.optimizerlib.NGOpt4,
    'NGOpt8': ng.optimization.optimizerlib.NGOpt8,
    'NGOpt10': ng.optimization.optimizerlib.NGOpt10,
    'NGOpt15': ng.optimization.optimizerlib.NGOpt15,
    'NGOpt16': ng.optimization.optimizerlib.NGOpt16,
    'NGOpt36': ng.optimization.optimizerlib.NGOpt36,
    'NGOpt39': ng.optimization.optimizerlib.NGOpt39,
    'NGOptRW': ng.optimization.optimizerlib.NGOptRW,
    'NGOptF': ng.optimization.optimizerlib.NGOptF,
    'NGOptF2': ng.optimization.optimizerlib.NGOptF2,
    'NGOptF3': ng.optimization.optimizerlib.NGOptF3,
    'NGOptF5': ng.optimization.optimizerlib.NGOptF5,
    'NGOpt': ng.optimization.optimizerlib.NGOpt,
    'Wiz': ng.optimization.optimizerlib.Wiz,
    'NgIoh': ng.optimization.optimizerlib.NgIoh,
    'NgIoh2': ng.optimization.optimizerlib.NgIoh2,
    'NgIoh3': ng.optimization.optimizerlib.NgIoh3,
    'NgIoh4': ng.optimization.optimizerlib.NgIoh4,
    'NgIohRW2': ng.optimization.optimizerlib.NgIohRW2,
    'NgIoh5': ng.optimization.optimizerlib.NgIoh5,
    'NgIoh6': ng.optimization.optimizerlib.NgIoh6,
    'SmoothDiscreteOnePlusOne': SmoothDiscreteOnePlusOne,
    'SmoothPortfolioDiscreteOnePlusOne': SmoothPortfolioDiscreteOnePlusOne,
    'SmoothDiscreteLenglerOnePlusOne': SmoothDiscreteLenglerOnePlusOne,
    'SmoothDiscreteLognormalOnePlusOne': SmoothDiscreteLognormalOnePlusOne,
    'SuperSmoothDiscreteLenglerOnePlusOne': SuperSmoothDiscreteLenglerOnePlusOne,
    'UltraSmoothDiscreteLenglerOnePlusOne': UltraSmoothDiscreteLenglerOnePlusOne,
    'SmoothLognormalDiscreteOnePlusOne': SmoothLognormalDiscreteOnePlusOne,
    'SmoothAdaptiveDiscreteOnePlusOne': SmoothAdaptiveDiscreteOnePlusOne,
    'SmoothRecombiningPortfolioDiscreteOnePlusOne': SmoothRecombiningPortfolioDiscreteOnePlusOne,
    'SmoothRecombiningDiscreteLanglerOnePlusOne': SmoothRecombiningDiscreteLanglerOnePlusOne,
    'UltraSmoothRecombiningDiscreteLanglerOnePlusOne': UltraSmoothRecombiningDiscreteLanglerOnePlusOne,
    'UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne': UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne,
    'UltraSmoothElitistRecombiningDiscreteLanglerOnePlusOne': UltraSmoothElitistRecombiningDiscreteLanglerOnePlusOne,
    'SuperSmoothElitistRecombiningDiscreteLanglerOnePlusOne': SuperSmoothElitistRecombiningDiscreteLanglerOnePlusOne,
    'SuperSmoothRecombiningDiscreteLanglerOnePlusOne': SuperSmoothRecombiningDiscreteLanglerOnePlusOne,
    'SuperSmoothRecombiningDiscreteLognormalOnePlusOne': SuperSmoothRecombiningDiscreteLognormalOnePlusOne,
    'SmoothElitistRecombiningDiscreteLanglerOnePlusOne': SmoothElitistRecombiningDiscreteLanglerOnePlusOne,
    'SmoothElitistRandRecombiningDiscreteLanglerOnePlusOne': SmoothElitistRandRecombiningDiscreteLanglerOnePlusOne,
    'SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne': SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne,
    'RecombiningDiscreteLanglerOnePlusOne': RecombiningDiscreteLanglerOnePlusOne,
    'RecombiningDiscreteLognormalOnePlusOne': RecombiningDiscreteLognormalOnePlusOne,
    'MaxRecombiningDiscreteLenglerOnePlusOne': MaxRecombiningDiscreteLenglerOnePlusOne,
    'MinRecombiningDiscreteLenglerOnePlusOne': MinRecombiningDiscreteLenglerOnePlusOne,
    'OnePtRecombiningDiscreteLenglerOnePlusOne': OnePtRecombiningDiscreteLenglerOnePlusOne,
    'TwoPtRecombiningDiscreteLenglerOnePlusOne': TwoPtRecombiningDiscreteLenglerOnePlusOne,
    'RandRecombiningDiscreteLenglerOnePlusOne': RandRecombiningDiscreteLenglerOnePlusOne,
    'RandRecombiningDiscreteLognormalOnePlusOne': RandRecombiningDiscreteLognormalOnePlusOne,
    'NgIoh7': ng.optimization.optimizerlib.NgIoh7,
    'NgDS11': ng.optimization.optimizerlib.NgDS11,
    'NgIoh11': ng.optimization.optimizerlib.NgIoh11,
    'NgIoh14': ng.optimization.optimizerlib.NgIoh14,
    'NgIoh13': ng.optimization.optimizerlib.NgIoh13,
    'NgIoh15': ng.optimization.optimizerlib.NgIoh15,
    'NgIoh12': ng.optimization.optimizerlib.NgIoh12,
    'NgIoh16': ng.optimization.optimizerlib.NgIoh16,
    'NgIoh17': ng.optimization.optimizerlib.NgIoh17,
    'NgDS': ng.optimization.optimizerlib.NgDS,
    'NgIoh21': ng.optimization.optimizerlib.NgIoh21,
    'NgDS2': ng.optimization.optimizerlib.NgDS2,
    'NGDSRW': ng.optimization.optimizerlib.NGDSRW,
    'NgIoh20': ng.optimization.optimizerlib.NgIoh20,
    'NgIoh19': ng.optimization.optimizerlib.NgIoh19,
    'NgIoh18': ng.optimization.optimizerlib.NgIoh18,
    'NgIoh10': ng.optimization.optimizerlib.NgIoh10,
    'NgIoh9': ng.optimization.optimizerlib.NgIoh9,
    'NgIoh8': ng.optimization.optimizerlib.NgIoh8,
    'MixDeterministicRL': MixDeterministicRL,
    'SpecialRL': SpecialRL,
    'NoisyRL1': NoisyRL1,
    'NoisyRL2': NoisyRL2,
    'NoisyRL3': NoisyRL3,
    'OnePointDE': OnePointDE,
    'ParametrizationDE': ParametrizationDE,
    'MiniDE': MiniDE,
    'MiniLhsDE': MiniLhsDE,
    'MiniQrDE': MiniQrDE,
    'AlmostRotationInvariantDEAndBigPop': AlmostRotationInvariantDEAndBigPop,
    'BPRotationInvariantDE': BPRotationInvariantDE,
    'MilliCMA': MilliCMA,
    'MicroCMA': MicroCMA,
    'FCMAs03': FCMAs03,
    'FCMAp13': FCMAp13,
    'ECMA': ECMA,
    'MetaModelDiagonalCMA': MetaModelDiagonalCMA,
    'MetaModelFmin2': MetaModelFmin2,
    'LSCMA': LSCMA,
    'HSCMA': HSCMA,
    'HSNeuralCMA': HSNeuralCMA,
    'HSSVMCMA': HSSVMCMA,
    'HSRFCMA': HSRFCMA,
    'HSMetaModel': HSMetaModel,
    'FastGADiscreteOnePlusOne': FastGADiscreteOnePlusOne,
    'DoubleFastGAOptimisticNoisyDiscreteOnePlusOne': DoubleFastGAOptimisticNoisyDiscreteOnePlusOne,
    'RecombiningGA': RecombiningGA,
    'RotatedRecombiningGA': RotatedRecombiningGA,
    'FastGAOptimisticNoisyDiscreteOnePlusOne': FastGAOptimisticNoisyDiscreteOnePlusOne,
    'FastGANoisyDiscreteOnePlusOne': FastGANoisyDiscreteOnePlusOne,
    'PortfolioOptimisticNoisyDiscreteOnePlusOne': PortfolioOptimisticNoisyDiscreteOnePlusOne,
    'PortfolioNoisyDiscreteOnePlusOne': PortfolioNoisyDiscreteOnePlusOne,
    'RecombiningOptimisticNoisyDiscreteOnePlusOne': RecombiningOptimisticNoisyDiscreteOnePlusOne,
    'RBO': RBO,
    'QRBO': QRBO,
    'MidQRBO': MidQRBO,
    'LBO': LBO,
    'IsoEMNA': IsoEMNA,
    'NaiveAnisoEMNA': NaiveAnisoEMNA,
    'AnisoEMNA': AnisoEMNA,
    'IsoEMNATBPSA': IsoEMNATBPSA,
    'NaiveIsoEMNATBPSA': NaiveIsoEMNATBPSA,
    'AnisoEMNATBPSA': AnisoEMNATBPSA,
    'NaiveAnisoEMNATBPSA': NaiveAnisoEMNATBPSA,
    'MetaCauchyRecentering': MetaCauchyRecentering,
    'ChainCMASQP': ChainCMASQP,
    'ChainDEwithR': ChainDEwithR,
    'ChainDEwithRsqrt': ChainDEwithRsqrt,
    'ChainDEwithRdim': ChainDEwithRdim,
    'ChainDEwithR30': ChainDEwithR30,
    'ChainDEwithLHS': ChainDEwithLHS,
    'ChainDEwithLHSsqrt': ChainDEwithLHSsqrt,
    'ChainDEwithLHSdim': ChainDEwithLHSdim,
    'ChainDEwithLHS30': ChainDEwithLHS30,
    'ChainDEwithMetaRecentering': ChainDEwithMetaRecentering,
    'ChainDEwithMetaRecenteringsqrt': ChainDEwithMetaRecenteringsqrt,
    'ChainDEwithMetaRecenteringdim': ChainDEwithMetaRecenteringdim,
    'ChainDEwithMetaRecentering30': ChainDEwithMetaRecentering30,
    'ChainBOwithMetaTuneRecentering': ChainBOwithMetaTuneRecentering,
    'ChainBOwithMetaTuneRecenteringsqrt': ChainBOwithMetaTuneRecenteringsqrt,
    'ChainBOwithMetaTuneRecenteringdim': ChainBOwithMetaTuneRecenteringdim,
    'ChainBOwithMetaTuneRecentering30': ChainBOwithMetaTuneRecentering30,
    'ChainDEwithMetaTuneRecentering': ChainDEwithMetaTuneRecentering,
    'ChainDEwithMetaTuneRecenteringsqrt': ChainDEwithMetaTuneRecenteringsqrt,
    'ChainDEwithMetaTuneRecenteringdim': ChainDEwithMetaTuneRecenteringdim,
    'ChainDEwithMetaTuneRecentering30': ChainDEwithMetaTuneRecentering30,
    'ChainBOwithR': ChainBOwithR,
    'ChainBOwithRsqrt': ChainBOwithRsqrt,
    'ChainBOwithRdim': ChainBOwithRdim,
    'ChainBOwithR30': ChainBOwithR30,
    'ChainBOwithLHS30': ChainBOwithLHS30,
    'ChainBOwithLHSsqrt': ChainBOwithLHSsqrt,
    'ChainBOwithLHSdim': ChainBOwithLHSdim,
    'ChainBOwithLHS': ChainBOwithLHS,
    'ChainBOwithMetaRecentering30': ChainBOwithMetaRecentering30,
    'ChainBOwithMetaRecenteringsqrt': ChainBOwithMetaRecenteringsqrt,
    'ChainBOwithMetaRecenteringdim': ChainBOwithMetaRecenteringdim,
    'ChainBOwithMetaRecentering': ChainBOwithMetaRecentering,
    'ChainPSOwithR': ChainPSOwithR,
    'ChainPSOwithRsqrt': ChainPSOwithRsqrt,
    'ChainPSOwithRdim': ChainPSOwithRdim,
    'ChainPSOwithR30': ChainPSOwithR30,
    'ChainPSOwithLHS30': ChainPSOwithLHS30,
    'ChainPSOwithLHSsqrt': ChainPSOwithLHSsqrt,
    'ChainPSOwithLHSdim': ChainPSOwithLHSdim,
    'ChainPSOwithLHS': ChainPSOwithLHS,
    'ChainPSOwithMetaRecentering30': ChainPSOwithMetaRecentering30,
    'ChainPSOwithMetaRecenteringsqrt': ChainPSOwithMetaRecenteringsqrt,
    'ChainPSOwithMetaRecenteringdim': ChainPSOwithMetaRecenteringdim,
    'ChainPSOwithMetaRecentering': ChainPSOwithMetaRecentering,
    'ChainCMAwithR': ChainCMAwithR,
    'ChainCMAwithRsqrt': ChainCMAwithRsqrt,
    'ChainCMAwithRdim': ChainCMAwithRdim,
    'ChainCMAwithR30': ChainCMAwithR30,
    'ChainCMAwithLHS30': ChainCMAwithLHS30,
    'ChainCMAwithLHSsqrt': ChainCMAwithLHSsqrt,
    'ChainCMAwithLHSdim': ChainCMAwithLHSdim,
    'ChainCMAwithLHS': ChainCMAwithLHS,
    'ChainCMAwithMetaRecentering30': ChainCMAwithMetaRecentering30,
    'ChainCMAwithMetaRecenteringsqrt': ChainCMAwithMetaRecenteringsqrt,
    'ChainCMAwithMetaRecenteringdim': ChainCMAwithMetaRecenteringdim,
    'ChainCMAwithMetaRecentering': ChainCMAwithMetaRecentering,
    'Zero': Zero,
    'StupidRandom': StupidRandom,
    'CauchyRandomSearch': CauchyRandomSearch,
    'RandomScaleRandomSearch': RandomScaleRandomSearch,
    'RandomScaleRandomSearchPlusMiddlePoint': RandomScaleRandomSearchPlusMiddlePoint,
    'RescaleScrHammersleySearch': RescaleScrHammersleySearch,
    'AvgHammersleySearch': AvgHammersleySearch,
    'AvgHammersleySearchPlusMiddlePoint': AvgHammersleySearchPlusMiddlePoint,
    'HullCenterHullAvgRandomSearch': HullCenterHullAvgRandomSearch,
    'AvgRandomSearch': AvgRandomSearch,
    'TEAvgScrHammersleySearchPlusMiddlePoint': TEAvgScrHammersleySearchPlusMiddlePoint,
    'TEAvgScrHammersleySearch': TEAvgScrHammersleySearch,
    'TEAvgRandomSearch': TEAvgRandomSearch,
    'TEAvgCauchyScrHammersleySearch': TEAvgCauchyScrHammersleySearch,
    'TEAvgLHSSearch': TEAvgLHSSearch,
    'TEAvgCauchyLHSSearch': TEAvgCauchyLHSSearch,
    'HullCenterHullAvgScrHaltonSearch': HullCenterHullAvgScrHaltonSearch,
    'HullCenterHullAvgScrHaltonSearchPlusMiddlePoint': HullCenterHullAvgScrHaltonSearchPlusMiddlePoint,
    'HullCenterHullAvgScrHammersleySearchPlusMiddlePoint': HullCenterHullAvgScrHammersleySearchPlusMiddlePoint,
    'HullCenterHullAvgLargeHammersleySearch': HullCenterHullAvgLargeHammersleySearch,
    'HullCenterHullAvgScrHammersleySearch': HullCenterHullAvgScrHammersleySearch,
    'HullCenterHullAvgCauchyScrHammersleySearch': HullCenterHullAvgCauchyScrHammersleySearch,
    'HullCenterHullAvgLHSSearch': HullCenterHullAvgLHSSearch,
    'HullCenterHullAvgCauchyLHSSearch': HullCenterHullAvgCauchyLHSSearch,
    'MetaNGOpt10': MetaNGOpt10,
    'NGOptSingle9': NGOptSingle9,
    'NGOptSingle16': NGOptSingle16,
    'NGOptSingle25': NGOptSingle25,
    'Noisy13Splits': Noisy13Splits,
    'NoisyInfSplits': NoisyInfSplits,
    'DiscreteNoisy13Splits': DiscreteNoisy13Splits,
    'DiscreteNoisyInfSplits': DiscreteNoisyInfSplits,
    'PCABO80': PCABO80,
    'PCABO95DoE20': PCABO95DoE20,
    'SparseDiscreteOnePlusOne': SparseDiscreteOnePlusOne,
    'HSDE': HSDE,
    'LhsHSDE': LhsHSDE,
    'FCarola6': FCarola6,
    'Carola11': Carola11,
    'Carola14': Carola14,
    'DS14': DS14,
    'Carola13': Carola13,
    'Carola15': Carola15,
    'NgIoh12b': ng.optimization.optimizerlib.NgIoh12b,
    'NgIoh13b': ng.optimization.optimizerlib.NgIoh13b,
    'NgIoh14b': ng.optimization.optimizerlib.NgIoh14b,
    'NgIoh15b': ng.optimization.optimizerlib.NgIoh15b,
    'NgDS3': NgDS3,
    'NgLn': NgLn,
    'CSEC': ng.optimization.optimizerlib.CSEC,
    'CSEC4': ng.optimization.optimizerlib.CSEC4,
    'CSEC5': ng.optimization.optimizerlib.CSEC5,
    'CSEC6': ng.optimization.optimizerlib.CSEC6,
    'CSEC7': ng.optimization.optimizerlib.CSEC7,
    'CSEC8': ng.optimization.optimizerlib.CSEC8,
    'CSEC9': ng.optimization.optimizerlib.CSEC9,
    'CSEC10': ng.optimization.optimizerlib.CSEC10
}
gradient_based__losses = {
    'MeanSquaredError': keras.losses.MeanSquaredError,
    'MeanAbsoluteError': keras.losses.MeanAbsoluteError,
    'MeanAbsolutePercentageError': keras.losses.MeanAbsolutePercentageError,
    'MeanSquaredLogarithmicError': keras.losses.MeanSquaredLogarithmicError,
    'SquaredHinge': keras.losses.SquaredHinge,
    'Hinge': keras.losses.Hinge,
    'CategoricalHinge': keras.losses.CategoricalHinge,
    'LogCosh': keras.losses.LogCosh,
    'Huber': keras.losses.Huber,
    'CategoricalCrossentropy': keras.losses.CategoricalCrossentropy,
    'SparseCategoricalCrossentropy': keras.losses.SparseCategoricalCrossentropy,
    'BinaryCrossentropy': keras.losses.BinaryCrossentropy,
    'KLDivergence': keras.losses.KLDivergence,
    'Poisson': keras.losses.Poisson,
    'CosineSimilarity': keras.losses.CosineSimilarity,
    'serialize': keras.losses.serialize,
    'deserialize': keras.losses.deserialize
}
parser.add_argument('--gradient_based__optimizer', type=str, help="Optimizer to be used for gradient based optimization. Possible choices: " + ", ".join(map(str, gradient_based__optimizers.keys())) + ".\n(default: Adam)", default="Adam")
parser.add_argument('--gradient_based__optimizer___learning_rate', type=float, help="Learning rate of the optimizer to be used for gradient based optimization. It has to be a float.\n(default: 1e-03)", default=1e-03)
parser.add_argument('--gradient_based__loss', type=str, help="Loss to be used for gradient based optimization. Possible choices: " + ", ".join(map(str, gradient_based__losses.keys())) + ".\n(default: CategoricalCrossentropy)", default="CategoricalCrossentropy")
parser.add_argument('--epochs', type=int, help="Training epochs. It has to be an int.\n(default: 1000)", default=1000)
parser.add_argument('--gradient_free__optimizer', type=str, help="Optimizer to be used for gradient free optimization. Possible choices: " + ", ".join(map(str, gradient_free__optimizers.keys())) + ".\n(default: CMA)", default="CMA")
args = parser.parse_args()
if args.split != 'auto' and args.split != 'train only':
    args.split = float(args.split)
if args.mean[0] != 'auto':
    for i in range(len(args.mean)):
        args.mean[i] = float(args.mean[i])
else:
    args.mean = args.mean[0]
if args.std[0] is None:
    args.std = None
else:
    if args.std[0] != 'auto':
        for i in range(len(args.std)):
            args.std[i] = float(args.std[i])
    else:
        args.std = args.std[0]
if args.height != 'auto':
    args.height = int(args.height)
if args.width != 'auto':
    args.width = int(args.width)
if args.batch_size != 'no batches':
    args.batch_size = int(args.batch_size)
if args.dom_rand__factors[0] is None:
    args.dom_rand__factors = None
if args.dom_rand__dist_pars[0] is None:
    args.dom_rand__dist_pars = None
else:
    if args.dom_rand__mode == 'uniform' or 'univariate normal':
        args.dom_rand__dist_pars = (args.dom_rand__dist_pars[:len(args.dom_rand__dist_pars)//2], args.dom_rand__dist_pars[len(args.dom_rand__dist_pars)//2:])
    if args.dom_rand__mode == 'triangular':
        args.dom_rand__dist_pars = tuple(args.dom_rand__dist_pars[i::len(args.dom_rand__dist_pars)//3] for i in range(len(args.dom_rand__dist_pars)//3))
    if args.dom_rand__mode == 'multivariate normal':
        n = len(args.dom_rand__dist_pars)
        dom_rand__dist_pars = args.dom_rand__dist_pars[:(n+n*n)], args.dom_rand__dist_pars[(n+n*n):]
if args.dom_rand__dist_pars___uniform[0] is None:
    args.dom_rand__dist_pars___uniform = None
else:
    args.dom_rand__dist_pars___uniform = (args.dom_rand__dist_pars___uniform[:len(args.dom_rand__dist_pars___uniform)//2], args.dom_rand__dist_pars___uniform[len(args.dom_rand__dist_pars___uniform)//2:])
if args.dom_rand__dist_pars___triangular[0] is None:
    args.dom_rand__dist_pars___triangular = None
else:
    args.dom_rand__dist_pars___triangular = tuple(args.dom_rand__dist_pars___triangular[i::len(args.dom_rand__dist_pars___triangular)//3] for i in range(len(args.dom_rand__dist_pars___triangular)//3))
if args.dom_rand__dist_pars___univariatenormal[0] is None:
    args.dom_rand__dist_pars___univariatenormal = None
else:
    args.dom_rand__dist_pars___univariatenormal = (args.dom_rand__dist_pars___univariatenormal[:len(args.dom_rand__dist_pars___univariatenormal)//2], args.dom_rand__dist_pars___univariatenormal[len(args.dom_rand__dist_pars___univariatenormal)//2:])
if args.dom_rand__dist_pars___multivariatenormal[0] is None:
    args.dom_rand__dist_pars___multivariatenormal = None
else:
    n = len(args.dom_rand__dist_pars___multivariatenormal)
    args.dom_rand__dist_pars___multivariatenormal = args.dom_rand__dist_pars___multivariatenormal[:(n+n*n)], args.dom_rand__dist_pars___multivariatenormal[(n+n*n):]
if args.adaptive_dom_rand__dist_ranges____low[0] is None:
    args.adaptive_dom_rand__dist_ranges____low = None
else:
    if args.dom_rand__mode == 'uniform' or 'univariate normal':
        args.adaptive_dom_rand__dist_ranges____low = (args.adaptive_dom_rand__dist_ranges____low[:len(args.adaptive_dom_rand__dist_ranges____low)//2], args.adaptive_dom_rand__dist_ranges____low[len(args.adaptive_dom_rand__dist_ranges____low)//2:])
    if args.dom_rand__mode == 'triangular':
        args.adaptive_dom_rand__dist_ranges____low = tuple(args.adaptive_dom_rand__dist_ranges____low[i::len(args.adaptive_dom_rand__dist_ranges____low)//3] for i in range(len(args.adaptive_dom_rand__dist_ranges____low)//3))
    if args.dom_rand__mode == 'multivariate normal':
        n = len(args.adaptive_dom_rand__dist_ranges____low)
        args.adaptive_dom_rand__dist_ranges____low = args.adaptive_dom_rand__dist_ranges____low[:(n+n*n)], args.adaptive_dom_rand__dist_ranges____low[(n+n*n):]
if args.adaptive_dom_rand__dist_ranges___uniform____low[0] is None:
    args.adaptive_dom_rand__dist_ranges___uniform____low = None
else:
    args.adaptive_dom_rand__dist_ranges___uniform____low = (args.adaptive_dom_rand__dist_ranges___uniform____low[:len(args.adaptive_dom_rand__dist_ranges___uniform____low)//2], args.adaptive_dom_rand__dist_ranges___uniform____low[len(args.adaptive_dom_rand__dist_ranges___uniform____low)//2:])
if args.adaptive_dom_rand__dist_ranges___triangular____low[0] is None:
    args.adaptive_dom_rand__dist_ranges___triangular____low = None
else:
    args.adaptive_dom_rand__dist_ranges___triangular____low = tuple(args.adaptive_dom_rand__dist_ranges___triangular____low[i::len(args.adaptive_dom_rand__dist_ranges___triangular____low)//3] for i in range(len(args.adaptive_dom_rand__dist_ranges___triangular____low)//3))
if args.adaptive_dom_rand__dist_ranges___univariatenormal____low[0] is None:
    args.adaptive_dom_rand__dist_ranges___univariatenormal____low = None
else:
    args.adaptive_dom_rand__dist_ranges___univariatenormal____low = (args.adaptive_dom_rand__dist_ranges___univariatenormal____low[:len(args.adaptive_dom_rand__dist_ranges___univariatenormal____low)//2], args.adaptive_dom_rand__dist_ranges___univariatenormal____low[len(args.adaptive_dom_rand__dist_ranges___univariatenormal____low)//2:])
if args.adaptive_dom_rand__dist_ranges___multivariatenormal____low[0] is None:
    args.adaptive_dom_rand__dist_ranges___multivariatenormal____low = None
else:
    n = len(args.adaptive_dom_rand__dist_ranges___multivariatenormal____low)
    args.adaptive_dom_rand__dist_ranges___multivariatenormal____low = args.adaptive_dom_rand__dist_ranges___multivariatenormal____low[:(n+n*n)], args.adaptive_dom_rand__dist_ranges___multivariatenormal____low[(n+n*n):]
if args.adaptive_dom_rand__dist_ranges____up[0] is None:
    args.adaptive_dom_rand__dist_ranges____up = None
else:
    if args.dom_rand__mode == 'uniform' or 'univariate normal':
        args.adaptive_dom_rand__dist_ranges____up = (args.adaptive_dom_rand__dist_ranges____up[:len(args.adaptive_dom_rand__dist_ranges____up)//2], args.adaptive_dom_rand__dist_ranges____up[len(args.adaptive_dom_rand__dist_ranges____up)//2:])
    if args.dom_rand__mode == 'triangular':
        args.adaptive_dom_rand__dist_ranges____up = tuple(args.adaptive_dom_rand__dist_ranges____up[i::len(args.adaptive_dom_rand__dist_ranges____up)//3] for i in range(len(args.adaptive_dom_rand__dist_ranges____up)//3))
    if args.dom_rand__mode == 'multivariate normal':
        n = len(args.adaptive_dom_rand__dist_ranges____up)
        adaptive_dom_rand__dist_ranges____up = args.adaptive_dom_rand__dist_ranges____up[:(n+n*n)], args.adaptive_dom_rand__dist_ranges____up[(n+n*n):]
if args.adaptive_dom_rand__dist_ranges___uniform____up[0] is None:
    args.adaptive_dom_rand__dist_ranges___uniform____up = None
else:
    args.adaptive_dom_rand__dist_ranges___uniform____up = (args.adaptive_dom_rand__dist_ranges___uniform____up[:len(args.adaptive_dom_rand__dist_ranges___uniform____up)//2], args.adaptive_dom_rand__dist_ranges___uniform____up[len(args.adaptive_dom_rand__dist_ranges___uniform____up)//2:])
if args.adaptive_dom_rand__dist_ranges___triangular____up[0] is None:
    args.adaptive_dom_rand__dist_ranges___triangular____up = None
else:
    args.adaptive_dom_rand__dist_ranges___triangular____up = tuple(args.adaptive_dom_rand__dist_ranges___triangular____up[i::len(args.adaptive_dom_rand__dist_ranges___triangular____up)//3] for i in range(len(args.adaptive_dom_rand__dist_ranges___triangular____up)//3))
if args.adaptive_dom_rand__dist_ranges___univariatenormal____up[0] is None:
    args.adaptive_dom_rand__dist_ranges___univariatenormal____up = None
else:
    args.adaptive_dom_rand__dist_ranges___univariatenormal____up = (args.adaptive_dom_rand__dist_ranges___univariatenormal____up[:len(args.adaptive_dom_rand__dist_ranges___univariatenormal____up)//2], args.adaptive_dom_rand__dist_ranges___univariatenormal____up[len(args.adaptive_dom_rand__dist_ranges___univariatenormal____up)//2:])
if args.adaptive_dom_rand__dist_ranges___multivariatenormal____up[0] is None:
    args.adaptive_dom_rand__dist_ranges___multivariatenormal____up = None
else:
    n = len(args.adaptive_dom_rand__dist_ranges___multivariatenormal____up)
    args.adaptive_dom_rand__dist_ranges___multivariatenormal____up = args.adaptive_dom_rand__dist_ranges___multivariatenormal____up[:(n+n*n)], args.adaptive_dom_rand__dist_ranges___multivariatenormal____up[(n+n*n):]
if args.adaptive_dom_rand__dist_ranges____low is not None and args.adaptive_dom_rand__dist_ranges____up is not None:
    adaptive_dom_rand__dist_ranges = [[[val_low, val_up] for val_low, val_up in zip(el_low, el_up)] for el_low, el_up in zip(args.adaptive_dom_rand__dist_ranges____low, args.adaptive_dom_rand__dist_ranges____up)]
else:
    adaptive_dom_rand__dist_ranges = None
if args.adaptive_dom_rand__dist_ranges___uniform____low is not None and args.adaptive_dom_rand__dist_ranges___uniform____up is not None:
    adaptive_dom_rand__dist_ranges___uniform = [[[val_low, val_up] for val_low, val_up in zip(el_low, el_up)] for el_low, el_up in zip(args.adaptive_dom_rand__dist_ranges___uniform____low, args.adaptive_dom_rand__dist_ranges___uniform____up)]
else:
    adaptive_dom_rand__dist_ranges___uniform = None
if args.adaptive_dom_rand__dist_ranges___triangular____low is not None and args.adaptive_dom_rand__dist_ranges___triangular____up is not None:
    adaptive_dom_rand__dist_ranges___triangular = [[[val_low, val_up] for val_low, val_up in zip(el_low, el_up)] for el_low, el_up in zip(args.adaptive_dom_rand__dist_ranges___triangular____low, args.adaptive_dom_rand__dist_ranges___triangular____up)]
else:
    adaptive_dom_rand__dist_ranges___triangular = None
if args.adaptive_dom_rand__dist_ranges___univariatenormal____low is not None and args.adaptive_dom_rand__dist_ranges___univariatenormal____up is not None:
    adaptive_dom_rand__dist_ranges___univariatenormal = [[[val_low, val_up] for val_low, val_up in zip(el_low, el_up)] for el_low, el_up in zip(args.adaptive_dom_rand__dist_ranges___univariatenormal____low, args.adaptive_dom_rand__dist_ranges___univariatenormal____up)]
else:
    adaptive_dom_rand__dist_ranges___univariatenormal = None
if args.adaptive_dom_rand__dist_ranges___multivariatenormal____low is not None and args.adaptive_dom_rand__dist_ranges___multivariatenormal____up is not None:
    adaptive_dom_rand__dist_ranges___multivariatenormal = [[[val_low, val_up] for val_low, val_up in zip(el_low, el_up)] for el_low, el_up in zip(args.adaptive_dom_rand__dist_ranges___multivariatenormal____low, args.adaptive_dom_rand__dist_ranges___multivariatenormal____up)]
else:
    adaptive_dom_rand__dist_ranges___multivariatenormal = None
if args.adaptive_dom_rand__dist_initials[0] is None:
    args.adaptive_dom_rand__dist_initials = None
else:
    if args.dom_rand__mode == 'uniform' or 'univariate normal':
        args.adaptive_dom_rand__dist_initials = (args.adaptive_dom_rand__dist_initials[:len(args.adaptive_dom_rand__dist_initials)//2], args.adaptive_dom_rand__dist_initials[len(args.adaptive_dom_rand__dist_initials)//2:])
    if args.dom_rand__mode == 'triangular':
        args.adaptive_dom_rand__dist_initials = tuple(args.adaptive_dom_rand__dist_initials[i::len(args.adaptive_dom_rand__dist_initials)//3] for i in range(len(args.adaptive_dom_rand__dist_initials)//3))
    if args.dom_rand__mode == 'multivariate normal':
        n = len(args.adaptive_dom_rand__dist_initials)
        adaptive_dom_rand__dist_initials = args.adaptive_dom_rand__dist_initials[:(n+n*n)], args.adaptive_dom_rand__dist_initials[(n+n*n):]
if args.adaptive_dom_rand__dist_initials___uniform[0] is None:
    args.adaptive_dom_rand__dist_initials___uniform = None
else:
    args.adaptive_dom_rand__dist_initials___uniform = (args.adaptive_dom_rand__dist_initials___uniform[:len(args.adaptive_dom_rand__dist_initials___uniform)//2], args.adaptive_dom_rand__dist_initials___uniform[len(args.adaptive_dom_rand__dist_initials___uniform)//2:])
if args.adaptive_dom_rand__dist_initials___triangular[0] is None:
    args.adaptive_dom_rand__dist_initials___triangular = None
else:
    args.adaptive_dom_rand__dist_initials___triangular = tuple(args.adaptive_dom_rand__dist_initials___triangular[i::len(args.adaptive_dom_rand__dist_initials___triangular)//3] for i in range(len(args.adaptive_dom_rand__dist_initials___triangular)//3))
if args.adaptive_dom_rand__dist_initials___univariatenormal[0] is None:
    args.adaptive_dom_rand__dist_initials___univariatenormal = None
else:
    args.adaptive_dom_rand__dist_initials___univariatenormal = (args.adaptive_dom_rand__dist_initials___univariatenormal[:len(args.adaptive_dom_rand__dist_initials___univariatenormal)//2], args.adaptive_dom_rand__dist_initials___univariatenormal[len(args.adaptive_dom_rand__dist_initials___univariatenormal)//2:])
if args.adaptive_dom_rand__dist_initials___multivariatenormal[0] is None:
    args.adaptive_dom_rand__dist_initials___multivariatenormal = None
else:
    n = len(args.adaptive_dom_rand__dist_initials___multivariatenormal)
    args.adaptive_dom_rand__dist_initials___multivariatenormal = args.adaptive_dom_rand__dist_initials___multivariatenormal[:(n+n*n)], args.adaptive_dom_rand__dist_initials___multivariatenormal[(n+n*n):]


# to use tensorboard. 
# %load_ext tensorboard

# for gpus.
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)


dataset = DataImage(
    data_path=args.data_path, 
    split=args.split, 
    transform=args.transform, 
    normalize=args.normalize, 
    mean=args.mean, 
    std=args.std, 
    resize=args.resize, 
    height=args.height, 
    width=args.width, 
    one_hot_encoding=args.one_hot_encoding, 
    name=args.name, 
    format=args.format,
    buffer_size=args.buffer_size,
    batch_size=args.batch_size
)

#print dataset info
# print(dataset)

trainset = dataset.get_set(split="train")
class_names = dataset.labels

# get some random training images and print them
# image_batch, label_batch = next(iter(trainset))
# plt.figure(figsize=(60, 60))
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     printable_object = dataset.print_item(image_batch['print_object'][i])
#     plt.imshow(printable_object)
#     label = label_batch[i].numpy().decode("utf-8")
#     plt.title(label, fontsize=20)
#     plt.axis("off")
# plt.show()


trainset = dataset.apply_one_hot_encoding(trainset)
out_dir = args.model_path

if args.domain_randomization:
    d_r = DomainRandomization_parameters(
        mode=args.dom_rand__mode,
        seed=args.dom_rand__seed,
        factors=None,
        params=args.dom_rand__dist_pars,
        optimize=args.adaptive_dom_rand,
        ranges=adaptive_dom_rand__dist_ranges,
        initials=args.adaptive_dom_rand__dist_initials
    )
    if args.dom_rand__factors is None:
        d_r.set_factors(factors=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    else:
        d_r.set_factors(factors=args.factors)
    if args.dom_rand__dist_pars___uniform is not None:
        d_r.set_uniform_params(
            lowers=args.dom_rand__dist_pars___uniform[0],
            uppers=args.dom_rand__dist_pars___uniform[1]
        )
    if args.dom_rand__dist_pars___triangular is not None:
        d_r.set_triangular_params(
            lowers=args.dom_rand__dist_pars___triangular[0],
            modes=args.dom_rand__dist_pars___triangular[1],
            uppers=args.dom_rand__dist_pars___triangular[2]
        )
    if args.dom_rand__dist_pars___univariatenormal is not None:
        d_r.set_univariatenormal_params(
            means=args.dom_rand__dist_pars___univariatenormal[0],
            variances=args.dom_rand__dist_pars___univariatenormal[1]
        )
    if args.dom_rand__dist_pars___multivariatenormal is not None:
        d_r.set_multivariatenormal_params(
            mean_vector=args.dom_rand__dist_pars___multivariatenormal[0],
            variancecovariance_matrix=args.dom_rand__dist_pars___multivariatenormal[1]
        )
    if adaptive_dom_rand__dist_ranges___uniform is not None:
        d_r.set_uniform__ranges(
            lowers=adaptive_dom_rand__dist_ranges___uniform[0],
            uppers=adaptive_dom_rand__dist_ranges___uniform[1]
        )
    if adaptive_dom_rand__dist_ranges___triangular is not None:
        d_r.set_triangular__ranges(
            lowers=adaptive_dom_rand__dist_ranges___triangular[0],
            modes=adaptive_dom_rand__dist_ranges___triangular[1],
            uppers=adaptive_dom_rand__dist_ranges___triangular[2]
        )
    if adaptive_dom_rand__dist_ranges___univariatenormal is not None:
        d_r.set_univariatenormal_ranges(
            means=adaptive_dom_rand__dist_ranges___univariatenormal[0],
            variances=adaptive_dom_rand__dist_ranges___univariatenormal[1]
        )
    if adaptive_dom_rand__dist_ranges___multivariatenormal is not None:
        d_r.set_multivariatenormal_ranges(
            mean_vector=adaptive_dom_rand__dist_ranges___multivariatenormal[0],
            variancecovariance_matrix=adaptive_dom_rand__dist_ranges___multivariatenormal[1]
        )
    if args.adaptive_dom_rand__dist_initials___uniform is not None:
        d_r.set_uniform__initials(
            lowers=args.adaptive_dom_rand__dist_initials___uniform[0],
            uppers=args.adaptive_dom_rand__dist_initials___uniform[1]
        )
    if args.adaptive_dom_rand__dist_initials___triangular is not None:
        d_r.set_triangular__initials(
            lowers=args.adaptive_dom_rand__dist_initials___triangular[0],
            modes=args.adaptive_dom_rand__dist_initials___triangular[1],
            uppers=args.adaptive_dom_rand__dist_initials___triangular[2]
        )
    if args.adaptive_dom_rand__dist_initials___univariatenormal is not None:
        d_r.set_univariatenormal_initials(
            means=args.adaptive_dom_rand__dist_initials___univariatenormal[0],
            variances=args.adaptive_dom_rand__dist_initials___univariatenormal[1]
        )
    if args.adaptive_dom_rand__dist_initials___multivariatenormal is not None:
        d_r.set_multivariatenormal_initials(
            mean_vector=args.adaptive_dom_rand__dist_initials___multivariatenormal[0],
            variancecovariance_matrix=args.adaptive_dom_rand__dist_initials___multivariatenormal[1]
        )
else:
    d_r = None

image_batch, label_batch = next(iter(trainset))
input_shape = image_batch['data'].shape[1:]

if args.neural_network == 'ResNet 1':
    model = ResNet1(len(class_names), input_shape=input_shape, field='data', domain_randomization=d_r)
if args.neural_network == 'ResNet 2.0':
    model = ResNet2__0(len(class_names), input_shape=input_shape, field='data', domain_randomization=d_r)
if args.neural_network == 'ResNet 2.1':
    model = ResNet2__1(len(class_names), input_shape=input_shape, field='data', domain_randomization=d_r)
if args.neural_network == 'ResNet 2.0.1':
    model = ResNet2__0__1(len(class_names), input_shape=input_shape, field='data', domain_randomization=d_r)
if args.neural_network == 'ResNet 2.1.1':
    model = ResNet2__1__1(len(class_names), input_shape=input_shape, field='data', domain_randomization=d_r)


class EarlyStoppingOnTarget(keras.callbacks.Callback):
    def __init__(self, monitor='accuracy', target=1.0, tensorboard_callback=None):
        super(EarlyStoppingOnTarget, self).__init__()
        self.monitor = monitor
        self.target = target
        self.tensorboard_callback = tensorboard_callback
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(f"Early stopping requires {self.monitor} available!", RuntimeWarning)
        if current is not None and current >= self.target:
            print(f"\n{self.monitor} target reached ({self.target}). Training stopped.")
            self.model.stop_training = True
        #if self.tensorboard_callback is not None:
        #self.tensorboard_callback.on_epoch_end(epoch, logs)

tb_callback = keras.callbacks.TensorBoard(log_dir=f'{out_dir}/tensorboard_log', histogram_freq=1)
es_callback = EarlyStoppingOnTarget(target=1.0, tensorboard_callback=tb_callback)


model.compile(
    optimizer=[(gradient_based__optimizers[args.gradient_based__optimizer], {"learning_rate":args.gradient_based__optimizer___learning_rate}),
               gradient_free__optimizers[args.gradient_free__optimizer]],
    loss=gradient_based__losses[args.gradient_based__loss],
    metrics=["accuracy"],
    run_eagerly=True,
)

tf.get_logger().setLevel('ERROR')
if args.adaptive_dom_rand:
    model.fit(trainset, epochs=args.epochs, callbacks=[tb_callback, es_callback], fverbose=2)
else:
    model.fit(trainset, epochs=args.epochs, callbacks=[tb_callback, es_callback])

#print model info
# model.summary()

model.save(out_dir)
keras.utils.plot_model(model, f'{out_dir}/ResNet.svg', show_shapes=True, show_layer_activations=True, show_layer_names=True)