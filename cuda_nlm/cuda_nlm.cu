/*
* cuda_nlm.cpp
*
*  Created on: March 17, 2015
*      Author: Denis Tananaev
*/

#include "cuda_nlm.h"
#include <cmath>
#include <cstdio>
#include "CMatrix.h"
#include <ctime>

using namespace std;


//Cuda_NLM_Naive
__global__ void cuda_NLM_filter(float* result, const float* input,float *gauss_lut, int x_size, int y_size,int window_radius,int patch_radius,float inv_sqr_sigma)
{
int x = blockDim.x * blockIdx.x + threadIdx.x;
int y = blockDim.y * blockIdx.y + threadIdx.y;

if(x>=x_size || y>=y_size)
return;

  //variables
 int gauss_lut_center = patch_radius;
	float sum = 0; 
    float new_value = 0;
	
	

	// window
    const int x1 = max(0,x-window_radius);
    const int y1 = max(0,y-window_radius);
    const int x2 = min(x_size-1,x+window_radius);
    const int y2 = min(y_size-1,y+window_radius);

	
	//patch comparing
    for( int ny = y1; ny <= y2; ++ny )
    for( int nx = x1; nx <= x2; ++nx )
    {
		float dist = 0;
  for( int ty = -patch_radius; ty <= patch_radius; ++ty )
  for( int tx = -patch_radius; tx <= patch_radius; ++tx )
  {
    // clamp coordinates
    int p1x = min(x_size-1,max(0,x+tx));
    int p1y = min(y_size-1,max(0,y+ty));
    int p2x = min(x_size-1,max(0,nx+tx));
    int p2y = min(y_size-1,max(0,ny+ty));

	//calculate distance between patches
    float tmp = input[p1y*x_size+p1x]-input[p2y*x_size+p2x];
    dist += tmp*tmp*gauss_lut[gauss_lut_center+tx]*gauss_lut[gauss_lut_center+ty];
  }

  //calculate weight of each patch
      float w = exp(-dist*inv_sqr_sigma);
  //weightet sum of the patches
      new_value += w*input[ny*x_size+nx];
  //normalizer
      sum+= w;
    }
	//synchronaise threads
	__syncthreads();

    result[y*x_size+x] = new_value/sum;

}

CMatrix<float> cuda_NLM_Naive(const CMatrix<float> &image, int window_radius, int patch_radius, float sqr_sigma){

 CMatrix<float> result(image.xSize(), image.ySize(), 0);
 float inv_sqr_sigma = 1. / sqr_sigma;


 //gauss lut
  float* gauss_lut = new float[2*patch_radius+1];
  float* gauss_lut_center = gauss_lut+patch_radius;
  for( int i = -patch_radius; i <= patch_radius; ++i )
    *(gauss_lut_center+i) = std::exp(-0.5*i*i/(patch_radius*patch_radius));

  //memory allocation of the device
  float* d_image;
  float* d_gauss_lut;
  float* d_result;

  cudaMalloc((void**)&d_image, image.size()*sizeof(float));
  cudaMalloc((void**)&d_result, image.size()*sizeof(float));
   cudaMalloc((void**)&d_gauss_lut, (2*patch_radius+1)*sizeof(float));

  // copy image to device
  cudaMemcpy(d_image, image.data(), 
              image.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy( d_gauss_lut, gauss_lut, 
             (2*patch_radius+1)*sizeof(float), cudaMemcpyHostToDevice);

  //kernel
  dim3 block(32, 32, 1);
  dim3 grid;
  grid.x = std::ceil(image.xSize()/(float)block.x);
  grid.y = std::ceil(image.ySize()/(float)block.y);
//naive filter
     cuda_NLM_filter<<<grid,block>>>(d_result, d_image, d_gauss_lut, image.xSize(), image.ySize(),window_radius,patch_radius, inv_sqr_sigma);


  // Copy result back
  cudaMemcpy( (void*)result.data(), (void*)d_result, 
              result.size()*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_image);
	cudaFree(d_result);
	cudaFree(d_gauss_lut);
  return result;
}
