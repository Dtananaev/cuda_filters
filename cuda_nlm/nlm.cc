/*
*     nlm.cc
*
*  Created on: March 17, 2015
*      Author: Denis Tananaev
*/
#include <cmath>
#include <iostream>
#include "CMatrix.h"
#include "cuda_nlm.h"
#include "cuda_nlm_smem.h"
#include <omp.h>
#include <timer_c.h>
#ifndef __linux__
#include <iso646.h>
#endif
#include "CTensor.h"

int main(int argc, char *argv[]){

    std::string fileNameInput;
	int window_radius=10;
	int patch_radius=3;
	float sigma=100;
    int n;

	if (argc==2){
		fileNameInput=argv[1];

	} else if (argc==3){
		fileNameInput=argv[1];
		patch_radius=atoi(argv[2]);

	} else if (argc==4){
		fileNameInput=argv[1];
		patch_radius=atoi(argv[2]);
		window_radius=atoi(argv[3]);
	} else if (argc==5){
		fileNameInput=argv[1];
		patch_radius=atoi(argv[2]);
		window_radius=atoi(argv[3]);  
		sigma=atoi(argv[4]); 
	}else{
		std::cout<<"!!!WRONG INPUT!!!"<<"\n";
		std::cout<<"Usage: cunlm inputfile  <path radius> <window radius> <sigma>"<<"\n";
		std::cout<<"The command should contain at least input file name. The default cunlm path_radius=3, window=10, sigma=100."<<"\n";
		return 0;    
	}
    CTensor<float> image_color;
	image_color.readFromPPM((fileNameInput+".ppm").c_str());

	CMatrix<float> red(image_color.xSize(),image_color.ySize()), filter_red(image_color.xSize(),image_color.ySize());
	CMatrix<float> green(image_color.xSize(),image_color.ySize()), filter_green(image_color.xSize(),image_color.ySize());
	CMatrix<float> blue(image_color.xSize(),image_color.ySize()), filter_blue(image_color.xSize(),image_color.ySize());

	CTensor<float> result_color(image_color.xSize(),image_color.ySize(),image_color.zSize(),0);
	CMatrix<float> Result(image_color.xSize(),image_color.ySize());

std::cout<<"Choose cuda nlm algorithm version:"<<"\n";
	std::cout<<"[1]- Naive nlm; [2]- nlm with shared memory"<<"\n";
	std::cin>>n;    
	if(n==1){
	 timer::start("gpu_naive");//t.start();
    filter_red=cuda_NLM_Naive(image_color.getMatrix(0),window_radius,patch_radius,sigma);
	std::cout<<"1/3 Done. "<<"\n";
	filter_green=cuda_NLM_Naive(image_color.getMatrix(1),window_radius,patch_radius,sigma);
	std::cout<<"2/3 Done. "<<"\n";
	filter_blue=cuda_NLM_Naive(image_color.getMatrix(2),window_radius,patch_radius,sigma);
	std::cout<<"3/3 Done. "<<"\n";
	  timer::stop("gpu_naive");
	}else if(n==2){
		timer::start("gpu_naive");//t.start();
    filter_red=cuda_NLM_Naive_smem(image_color.getMatrix(0),window_radius,patch_radius,sigma);
	std::cout<<"1/3 Done. "<<"\n";
	filter_green=cuda_NLM_Naive_smem(image_color.getMatrix(1),window_radius,patch_radius,sigma);
	std::cout<<"2/3 Done. "<<"\n";
	filter_blue=cuda_NLM_Naive_smem(image_color.getMatrix(2),window_radius,patch_radius,sigma);
	std::cout<<"3/3 Done. "<<"\n";
	  timer::stop("gpu_naive");
	
	}else{
		std::cout<<"!!!WRONG INPUT!!!"<<"\n";
		return 0;
	}

	 

    result_color.putMatrix(filter_red,0);
	result_color.putMatrix(filter_green,1);
	result_color.putMatrix(filter_blue,2);
	result_color.writeToPPM((fileNameInput+"_cunlm.ppm").c_str());


	  timer::printToScreen(); timer::reset();


  return 0;
}

