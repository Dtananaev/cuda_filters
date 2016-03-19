/**
 * ====================================================================
 *
 * Author: Nikolaus Mayer, 2014  (mayern@informatik.uni-freiburg.de)
 *
 * ====================================================================
 *
 * Display CMatrix and CTensor images (using CImg)
 *
 * ====================================================================
 *
 * Setup: If not automatically done so by your compiler, include the
 * ====== following flags into your compilation string:
 *        -stc=c++11 -pthread
 *        and the following in your linker command:
 *        -lX11 -pthread
 *
 *        In case you are using CMake, include the packages "X11" and
 *        "Threads". Caution, the Threads package has special library
 *        variables.
 *
 * ====================================================================
 *
 * Usage:
 * ======
 *
 * +------------------------------------------------------------------+
 * | #include "CTensor"                                               |
 * | #include "ImageDisplay.h"                                        |
 * |                                                                  |
 * | int main( int argv, char** argv )                                |
 * | {                                                                |
 * |   /// Setup ImageDisplay instance                                |
 * |   ImageDisplay::ImageDisplay my_display;                         |
 * |                                                                  |
 * |   /// Load some CTensor (or CMatrix) image                       |
 * |   CTensor<float> ctensor_image;                                  |
 * |   ctensor_image.readFromPPM("imagefile.ppm");                    |
 * |                                                                  |
 * |   /// Display the image                                          |
 * |   my_display.Display(ctensor_image, "Awesome Image");            |
 * |                                                                  |
 * |   /// At the end of its scope, the ImageDisplay instance will    |
 * |   /// block until you close all of its windows.                  |
 * |   /// Alternatively, you can call its CloseAll() method (then    |
 * |   /// it has no more open windows and will not block).           |
 * | }                                                                |
 * +------------------------------------------------------------------+
 *
 * !WARNING! It is important to take care that any ImageDisplay 
 * ========= instances are destroyed BEFORE THE APPLICATION ENDS.
 *          
 *           Either set them up as local variables (which will lead to
 *           them being destroyed when their scope ends), or "delete"
 *           your global instance before main() terminates.
 *          
 *           If this is not done correctly, the CImg library can get 
 *           caught in a systemside threading lock, forcing you to kill
 *           your application manually.
 *
 * ====================================================================
 */

#ifndef IMAGEDISPLAY_H__
#define IMAGEDISPLAY_H__


/// System/STL
#include <map>      // std::map
#include <string>
#include <utility>  // std::pair
/// Local files
#include "CImg.h"
#include "CMatrix.h"
#include "CTensor.h"

using namespace cimg_library;


namespace ImageDisplay_AgnosticCImg {

  /// /////////////////////////////////////////////////////////////////
  /// Helper classes
  /// /////////////////////////////////////////////////////////////////
  
  /**
   * Base class that unifies CImg instances with different types
   */
  class AgnosticCImgBase {
  public:
    /// Destructor
    virtual ~AgnosticCImgBase(){};
    virtual void get( void*& arg ) = 0;
  };

  /**
   * Store a CImg instance
   */
  template <typename T>
  class AgnosticCImg : public AgnosticCImgBase {
  public:
    /// Constructor
    AgnosticCImg( const CImg<T>& cimg )
    {
      /// Deep copy
      m_cimg = CImg<T>(cimg, false);
    }

    /// Destructor
    virtual ~AgnosticCImg(){};

    /**
     * Yield address of m_cimg
     */
    virtual void get( void*& arg )
    {
      arg = (void*)(&m_cimg);
    };

  private:
    CImg<T> m_cimg;
  };

}  // namespace ImageDisplay_AgnosticCImg 


using namespace ImageDisplay_AgnosticCImg;


namespace ImageDisplay {

  /// /////////////////////////////////////////////////////////////////
  /// ImageDisplay class
  /// /////////////////////////////////////////////////////////////////

  class ImageDisplay 
  {

  public:
    
    /**
     * Constructor
     *
     * @param print_information Iff TRUE, the instance will print all sorts of more or less useful information to stdout
     */
    ImageDisplay( bool print_information=true,
                  const std::string& instance_name="ImageDisplay"
                )
      : m_print_information(print_information),
        m_instance_name(instance_name)
    { };


    /// Destructor
    ~ImageDisplay()
    {
      bool print = false;

      bool a_window_is_still_open = false;
      for ( auto& item : m_map ) {
        if ( !item.second->second->is_closed() ) {
          a_window_is_still_open = true;
          break;
        }
      }

      if ( m_print_information and a_window_is_still_open ) {
        print = true;
        std::cout << m_instance_name
                  << ": Waiting until all my windows have"
                  << " been closed...\n"
                  << std::flush;
      }

      /// Stay alive until the user has closed all windows managed by
      /// this instance
      if ( a_window_is_still_open ) do {
        //std::cout << "CHECKING" << std::endl;
        a_window_is_still_open = false;
        //size_t c = 0;
        for ( auto& item : m_map ) {
          if ( !item.second->second->is_closed() ) {
            a_window_is_still_open = true;
            //std::cout << item.first << std::endl;
            //item.second->second->wait();
            CImgDisplay::wait_all();
            break;
            //++c;
          }
        }
        //std::cout << c << std::endl;
      } while ( a_window_is_still_open );

      if ( m_print_information and print )
        std::cout << m_instance_name
                  << ": All my windows are gone, exiting...\n";

      /// Tidy up
      for ( auto& item: m_map ) {
        delete item.second->second;  // CImgDisplay*
        delete item.second->first;  // AgnosticCImgBase*
        delete item.second;  // std::pair<AgnosticCImgBase*, CImgDisplay*>*
      }
    };


    /**
     * Display a CTensor image in a named window
     *
     * @param ctensor CTensor image
     * @param windowtitle Window title
     */
    template <typename T>
    void Display( const CTensor<T>& ctensor,
                  const std::string windowtitle 
                )
    {
      /// Setup a temporary CImg instance to copy the input CTensor 
      CImg<T> tmp( ctensor.data(),  /// <-- Identical memory layout!
                   ctensor.xSize(),
                   ctensor.ySize(),
                   1,
                   ctensor.zSize(),
                   false  /// copy data, don't share it
                 );
      /// Copy the temporary CImg into a AgnosticCImgBase so we can
      /// push it into our STL map container
      AgnosticCImg<T>* first = 
          new ImageDisplay_AgnosticCImg::AgnosticCImg<T>(tmp);
      /// Our CImgDisplay needs a pointer to the AgnosticCImg's data
      CImg<T>* tmp_cimg_ptr(0);
      first->get((void*&)(tmp_cimg_ptr));
      
      /// If the instance already has a display with title "windowtitle",
      /// that display is updated to show the new image
      if ( m_map.count( windowtitle ) > 0 ) {
        /// TODO destroy old image
        m_map[windowtitle]->second->display(*tmp_cimg_ptr);
        delete m_map[windowtitle]->first;
        m_map[windowtitle]->first = first;

        if ( m_print_information ) 
          std::cout << m_instance_name
                    << ": Updated display '" 
                    << windowtitle
                    << "'.\n";
                  
      } else {      
        /// else a new display is created and added to the map
        CImgDisplay* second = 
            new CImgDisplay(*tmp_cimg_ptr, windowtitle.c_str());
        m_map[windowtitle] =  
            new std::pair<ImageDisplay_AgnosticCImg::AgnosticCImgBase*, 
                          CImgDisplay*>(first, second);

        if ( m_print_information ) 
          std::cout << m_instance_name
                    << ": Created new display '" 
                    << windowtitle
                    << "'.\n";
      }
    }

    /**
     * Display a CTensor image in a named window
     *
     * @param windowtitle Window title
     * @param ctensor CTensor image
     */
    template <typename T>
    void Display( const std::string windowtitle, 
                  const CTensor<T>& ctensor
                )
    {
      Display(ctensor, windowtitle);
    }

    /**
     * Display a CMatrixs image in a named window
     *
     * @param cmatrix CMatrix image
     * @param windowtitle Window title
     */
    template <typename T>
    void Display( const CMatrix<T>& cmatrix,
                  const std::string windowtitle 
                )
    {
      /// Setup a temporary one-layer CTensor. The first layer will
      /// be set to the cmatrix.
      CTensor<T> tmp_ctensor1d(cmatrix.xSize(), cmatrix.ySize(), 1);
      /// Casting away const-ness is not nice, but we'll just trust
      /// the CTensor to not do anything bad to the CMatrix.
      tmp_ctensor1d.putMatrix(const_cast<CMatrix<T>&>(cmatrix), 0);
      Display(tmp_ctensor1d, windowtitle);
    }

    /**
     * Display a CMatrixs image in a named window
     *
     * @param windowtitle Window title
     * @param cmatrix CMatrix image
     */
    template <typename T>
    void Display( const std::string windowtitle, 
                  const CMatrix<T>& cmatrix
                )
    {
      Display(cmatrix, windowtitle);
    }



  private:

    bool m_print_information;
    std::string m_instance_name;

    /// Map window titles to CImg/CImgDisplay pairs
    std::map< std::string,
              std::pair< ImageDisplay_AgnosticCImg::AgnosticCImgBase*,
                         CImgDisplay*
                       >*
            > m_map;
                       
  };




}  // namespace ImageDisplay




#endif  // IMAGEDISPLAY_H__

