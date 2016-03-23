#include "CTensor.h"
#include "timer.h"
#include <string>

typedef uchar4 Color;

/*
 *  CTensor stores the three colors separately in a 3D array
 *  We want to store the image in a 2D matrix where each element holds the RGB value
 */
void CTensorToColorCMatrix(CMatrix<Color>& out, const CTensor<unsigned char>& in)
{
  out.setSize(in.xSize(), in.ySize());
  for( int y = 0; y < out.ySize(); ++y )
  for( int x = 0; x < out.xSize(); ++x )
  {
    out(x,y).x = in(x,y,0); // R
    out(x,y).y = in(x,y,1); // G
    out(x,y).z = in(x,y,2); // B
  }
}


/*
 *  The inverse function to CTensorToColorMatrix()
 */
void ColorCMatrixToCTensor(CTensor<unsigned char>& out, const CMatrix<Color>& in)
{
  out.setSize(in.xSize(), in.ySize(), 3);
  for( int y = 0; y < out.ySize(); ++y )
  for( int x = 0; x < out.xSize(); ++x )
  {
    out(x,y,0) = in(x,y).x; // R
    out(x,y,1) = in(x,y).y; // G
    out(x,y,2) = in(x,y).z; // B
  }
}


/*
 *  Inverts a RGB color image
 *
 *  img     The input and output image
 *  x_size  image width in px
 *  y_size  image height in px
 *  pitch   row size in bytes
 */
__global__ void invert_kernel( Color* img, int x_size, int y_size, int pitch )
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if( x >= x_size || y >= y_size )
    return;

  Color* ptr = (Color*)((char*)img + y * pitch) + x;
  Color color = *ptr;
  color.x = 255-color.x;
  color.y = 255-color.y;
  color.z = 255-color.z;
  *ptr = color;
}


int main( int argc, char** argv )
{

	std::string fileNameInput;

	if (argc==2){
		fileNameInput=argv[1];
	}else{
		std::cout<<"!!!WRONG INPUT!!!"<<"\n";
		std::cout<<"Usage: cuinv inputfile "<<"\n";
		std::cout<<"The command should contain  input file name."<<"\n";
		return 0;    
	}

  // Read the image from disk R G B
  CTensor<unsigned char> tmp;
  tmp.readFromPPM((fileNameInput+".ppm").c_str());

  // Store the image in an appropriate format
  CMatrix<Color> img;
  CTensorToColorCMatrix(img, tmp);

  // Copy image to the device
  Color* d_img;
  cudaMalloc((void**)&d_img, sizeof(Color)*img.size());
  cudaMemcpy((void*)d_img, (void*)img.data(), sizeof(Color)*img.size(), cudaMemcpyHostToDevice);

  // Setup kernel launch
  dim3 block(16,16,1);
  dim3 grid;
  grid.x = std::ceil( img.xSize()/(float)block.x );
  grid.y = std::ceil( img.ySize()/(float)block.y );
  timer::start("uncoalesced global memory access");
  invert_kernel<<<grid,block>>>(d_img, img.xSize(), img.ySize(), sizeof(Color)*img.xSize());
  timer::stop("uncoalesced global memory access");

  // Copy result back
  cudaMemcpy((void*)img.data(), (void*)d_img, sizeof(Color)*img.size(), cudaMemcpyDeviceToHost);

  // Write inverted image to the disk
  ColorCMatrixToCTensor(tmp, img);
  tmp.writeToPPM((fileNameInput+"_cuinv.ppm").c_str());


  Color* d_img_aligned;
  size_t pitch;
  /*
   *
   * Allocate memory here in d_img_aligned using cudaMallocPitch()
   *
   */
	
cudaMallocPitch((void**)&d_img_aligned, &pitch, sizeof(Color)*img.xSize(),img.ySize());

  std::cout << "Image row size " << sizeof(Color)*img.xSize() << std::endl;
  std::cout << "Pitch          " << pitch << std::endl;
  /*
   *
   * Copy the image in 'img' to the device with cudaMemcpy2D() 
   * Make sure you use the right pitch for 'img' and 'd_img_aligned'
   *
   */
 
cudaMemcpy2D((void*)d_img_aligned, pitch, 
			(void*)img.data(), sizeof(Color)*img.xSize(),sizeof(Color)*img.xSize(),img.ySize() , cudaMemcpyHostToDevice);

//timer
  timer::start("coalesced global memory access");
  invert_kernel<<<grid,block>>>(d_img_aligned, img.xSize(), img.ySize(), pitch);
  timer::stop("coalesced global memory access");

  /* Copy the image back to the host with cudaMemcpy2D() */
 cudaMemcpy2D((void*)img.data(),sizeof(Color)*img.xSize(),
				(void*)d_img_aligned,pitch,sizeof(Color)*img.xSize(),img.ySize(), cudaMemcpyDeviceToHost);

 

  // Write the inverted inverted image to the disk
  ColorCMatrixToCTensor(tmp, img);
  tmp.writeToPPM((fileNameInput+"_cuinv_restored.ppm").c_str());

  timer::printToScreen();


  cudaFree((void*)d_img);
  cudaFree((void*)d_img_aligned);
  return 0;
}
