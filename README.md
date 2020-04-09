This repository provides Matlab and Python codes that implement a 3D refractive index reconstruction algorithm using 3D DPC through focus intensity measurements. Images should be captured with an LED array illumination or other switchable light sources, generating three(or more) half circular(or half annular) patterns. The 3D Weak Object Transfer Functions(WOTFs) are calculated according to the source patterns, pupil function, and the defocus step size. Finally, 3D refractive index are solved after a 3D deconvolution process. As in 2D DPC case, a least squares algorithm with Tikhonov regularization is implemented. Alternatively, deconvolution with total variation(TV) regularization and non-negativity constraint further mitigates artifacts based on a-priori knowledge of the object. GPU codes are available if hardware allows, which largely reduces computation time. In Python codes, we use ArrayFire library (https://arrayfire.com/) and its python wrapper (https://github.com/arrayfire/arrayfire-python) for GPU processing implementation.

**Run the "main_3ddpc.m" under matlab_code folder, or open the "main_3ddpc.ipynb" jupyter notebook under python_code folder.

**Example dataset can be downloaded [here](https://drive.google.com/drive/folders/1U1JBysZeTS_YpybkYdcIPl6u968ZscLF?usp=sharing)

Please cite as:
[1] M. Chen, L. Tian and L. Waller, 3D differential phase contrast microscopy, Biomed. Opt. Express 7(10), 3940-3950 (2016).
[2] M. Chen and L. Waller, 3D Phase Microscopy with Coded Illumination, Biomed. Optics in the Life Sciences Congress:Novel Techniques in Microscopy, NM2C.2 (2017).
