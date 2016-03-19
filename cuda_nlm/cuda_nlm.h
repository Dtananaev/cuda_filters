/*
* cuda_nlm.cpp
*
*  Created on: March 17, 2015
*      Author: Denis Tananaev
*/
#pragma once
#ifndef CUDA_NLM_H_
#define CUDA_NLM_H_
#include <cstdio>
#include "CMatrix.h"

CMatrix<float> cuda_NLM_Naive(const CMatrix<float> &image, int window_radius, int patch_radius, float sqr_sigma);


#endif // CUDA_NLM_H_
