Computer vision: GPU implementation of the filters for Nvidia CUDA
====================================================
All filters can process only files in ppm format.

[![Build Status](https://travis-ci.org/Dtananaev/cuda_filters.svg?branch=master)](https://travis-ci.org/Dtananaev/cuda_filters)
[![BSD2 License](http://img.shields.io/badge/license-BSD2-brightgreen.svg)](https://github.com/Dtananaev/cuda_filters/blob/master/LICENSE.md) 
     

It contains:

* cuda_nlm - non-local mean filter implementation
<p align="center">
  <img src="https://github.com/Dtananaev/cuda_filters/blob/master/pictures/tower_gaussn.jpg" width="350"/>
  <img src="https://github.com/Dtananaev/cuda_filters/blob/master/pictures/tower_gaussn_cunlm.jpg" width="350"/>
</p>
      * To install:
           * install cuda nvcc compiler
           * cd ../cuda_filters/cuda_nlm
           * make
      * To run:
           ./cunlm filename  \<path radius\> \<window radius\> \<sigma\> 
* cuda_inv - inverse of the picture color
<p align="center">
  <img src="https://github.com/Dtananaev/cuda_filters/blob/master/pictures/lena.jpg" width="350"/>
  <img src="https://github.com/Dtananaev/cuda_filters/blob/master/pictures/lena_inverted.jpg" width="350"/>
</p>
      * To install:
           * install cuda nvcc compiler
           * cd ../cuda_filters/cuda_inv
           * make
      * To run:
           ./cuinv filename 
* cuda_median - 3x3 median filter which is denoise "salt and pepper" type of noise
<p align="center">
  <img src="https://github.com/Dtananaev/cuda_filters/blob/master/pictures/balloons_noisy.jpg" width="350"/>
  <img src="https://github.com/Dtananaev/cuda_filters/blob/master/pictures/balloons_noisy_cumedian.jpg" width="350"/>
</p>
      * To install:
           * install cuda nvcc compiler
           * cd ../cuda_filters/cuda_median
           * make
      * To run:
           ./cumed filename \<number of application of the filter\>
