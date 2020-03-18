# demo-resLF
Reimplementation of resLF(CVPR2019): Residual Networks for Light Field Image Super-Resolution
(http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Residual_Networks_for_Light_Field_Image_Super-Resolution_CVPR_2019_paper.pdf)

## Motivation
The official code released by authors is at (https://github.com/shuozh/resLF), which is not testable because of the unavailable test images.

I notice that in the official code the test images are in the png format. However, some avaiable light field(LF) test images are in the mat format.

On the one hand, we transform the LF data into png format but we got the same issue as(https://github.com/shuozh/resLF/issues/8)

On the other hand, we downsample the test images(in the mat format) in Matlab to get the low-resolution inputs and write our own script for testing.
However, the outputs we get contain artifacts and do not reflect the results in the paper.

## Useage

Please download the pre-trained model in the folder.
```
eval.py [-image_path] [--model] [--scale] [--view_n] [--interpolation] [--gpu_no]
```
We only list the results of bicubic interpolation (x2) on the Buddha and Mona.

## Performance(PSNR/SSIM) on Budda and Mona
  
| Name   | Avg        |   Max      |  Min        |
| -------|:-----------:|:-----------:|:-----------:|
| Buddha | 39.33/0.9825 | 40.66/0.9866 | 38.29/0.9744|
| Mona   | 40.89/0.9879 | 41.93/0.9907 | 38.96/0.9809|

