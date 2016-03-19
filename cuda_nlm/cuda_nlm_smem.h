/*
* cuda_nlm_smem.h
*
*  Created on: March 17, 2015
*      Author: Denis Tananaev
*/

//cuda NLM with shared memory


#pragma once
#ifndef CUDA_NLM_SMEM_H_
#define CUDA_NLM_SMEM_H_
#include <cstdio>
#include "CMatrix.h"

CMatrix<float> cuda_NLM_Naive_smem(const CMatrix<float> &image, int window_radius, int patch_radius, float sqr_sigma);


#endif // CUDA_NLM_SMEM_H_
