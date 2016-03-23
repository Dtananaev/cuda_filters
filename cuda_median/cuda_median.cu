/*
* cuda_median.cu
*
*  Created on: February 15, 2015
*      Author: Denis Tananaev
*/


#include "CMatrix.h"
#include <stdio.h>
#include "CTensor.h"

__global__ void filter_kernel( float* result, float* img,int n,int m)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  float filter[9];

  int ym1,yp1,xm1,xp1;
  if( y == 0 ) ym1=y; else ym1 = y-1;

  // upper row
  filter[0] = img[ym1*n + (x-1)];
  filter[1] = img[ym1*n + (x)];
  filter[2] = img[ym1*n + (x+1)];

  // middle row
  filter[3] = img[(y)*n + (x-1)];
  filter[4] = img[(y)*n + (x)];
  filter[5] = img[(y)*n + (x+1)];

  // below row
  filter[6] = img[(y+1)*n + (x-1)];
  filter[7] = img[(y+1)*n + (x)];
  filter[8] = img[(y+1)*n + (x+1)];

  // filter
  for(int i = 0; i < 9; ++i)
  for(int j = 0; j < 8-i; ++j){
    if( filter[j] > filter[j+1] ){
      float tmp = filter[j];
      filter[j] = filter[j+1];
      filter[j+1] = tmp;
    }
  }

  result[y*n + x] = filter[4];

}
		
void apply_filter(CMatrix<float> img, CMatrix<float> & result){

    int n=img.xSize();
	int m=img.ySize();
  float * d_image, * d_result;
//Memory allocation for CUDA_device image and result

  cudaMalloc( (void**)& d_image, sizeof(float)*img.xSize()*img.ySize());
  cudaMalloc( (void**)& d_result, sizeof(float)*img.xSize()*img.ySize());

 // Copy input image to CUDA_device
 cudaMemcpy(d_image,img.data(),sizeof(float)*img.xSize()*img.ySize(),cudaMemcpyHostToDevice);
  dim3 block(16,16,1);
  dim3 grid;
  grid.x = std::ceil( img.xSize()/(float)block.x );
  grid.y = std::ceil( img.ySize()/(float)block.y );

filter_kernel<<<grid,block>>>(d_result,d_image,n,m);

 //CMatrix<float> result(img.xSize(),img.ySize());
  // Copy to host
  cudaMemcpy(result.data(), d_result, sizeof(float)*result.xSize()*result.ySize(), cudaMemcpyDeviceToHost);

  	cudaFree(d_image);
	cudaFree(d_result);
}


int main( int argc, char** argv )
{
    std::string fileNameInput;
	int n=1;

	if (argc==2){
		fileNameInput=argv[1];

	}else if (argc==3){
		fileNameInput=argv[1];
		n=atoi(argv[2]);
	}else{
		std::cout<<"!!!WRONG INPUT!!!"<<"\n";
		std::cout<<"Usage: cumedian inputfile <number of filter application>"<<"\n";
		std::cout<<"The command should contain input file name. By default filter appy n=1 times"<<"\n";
		return 0;    
	}

	 CTensor<float> image_color;
	image_color.readFromPPM((fileNameInput+".ppm").c_str());
	for(int i=0;i<n; i++){
   

	CMatrix<float>  filter_red(image_color.xSize(),image_color.ySize());
	CMatrix<float>  filter_green(image_color.xSize(),image_color.ySize());
	CMatrix<float>  filter_blue(image_color.xSize(),image_color.ySize());

	CTensor<float> result_color(image_color.xSize(),image_color.ySize(),image_color.zSize(),0);

   apply_filter(image_color.getMatrix(0),filter_red);

   apply_filter(image_color.getMatrix(1),filter_green);

   apply_filter(image_color.getMatrix(2),filter_blue);


	image_color.putMatrix(filter_red,0);
	image_color.putMatrix(filter_green,1);
	image_color.putMatrix(filter_blue,2);
	}

 image_color.writeToPPM((fileNameInput+"_cumedian.ppm").c_str());

  return 0;
}

