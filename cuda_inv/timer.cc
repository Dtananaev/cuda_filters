#include "timer.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <map>

namespace 
{
  using namespace timer;

  struct Timer
  {
    cudaEvent_t start;
    cudaEvent_t stop;
    bool running;
    float elapsedTime;
  };

  float dummy_elapsedTime;

  //! map holding all the timers
  std::map<std::string, Timer> timers;

  Format bestFormat(float elapsedTime)
  {
    if( elapsedTime < 1000.f )
      return MILLISEC;
    else if( elapsedTime < 60000.f )
      return SEC;
    else if( elapsedTime < 60*60000.f )
      return MIN;
    else 
      return HOUR;
  }


  std::string elapsedTimeToString( float elapsedTime, Format format,
                                   int width = 12, int precision = 3 )
  {
    std::stringstream sstr;


    if( format == AUTO )
      format = bestFormat(elapsedTime);
    switch( format )
    {
    case HOUR:
      sstr << std::setw(width) << std::fixed << std::setprecision(precision);
      sstr << elapsedTime/(60*60*1000.f) << " h";
      break;
    case MIN:
      sstr << std::setw(width) << std::fixed << std::setprecision(precision);
      sstr << elapsedTime/(60*1000.f) << " min";
      break;
    case SEC:
      sstr << std::setw(width) << std::fixed << std::setprecision(precision);
      sstr << elapsedTime/(1000.f) << " sec";
      break;
    case HHMMSS:
      sstr << std::setw(width);
      {
        int hh,mm,ss;
        hh = elapsedTime/(60*60*1000.f);
        elapsedTime -= hh*(60*60*1000.f);
        mm = elapsedTime/(60*1000.f);
        elapsedTime -= mm*(60*1000.f);
        ss = elapsedTime/(1000.f);
        sstr << hh << ":" << mm << ":" << ss;
      }
      break;
    default:
      sstr << std::setw(width) << std::fixed << std::setprecision(precision);
      sstr << elapsedTime << " msec";
    }
    return sstr.str();
  }


  struct compare_time
  {
    bool operator()(const std::pair<std::string,float>& a,
                    const std::pair<std::string,float>& b) const
    {
      return a.second < b.second;
    }
  };
  
  
  void printTimers( std::ostream& stream, const std::map<std::string,Timer>& timers, 
                    Format format, Sorting sorting, bool ascending )
  {
    typedef std::map<std::string, Timer>::const_iterator map_const_iter;

    std::vector<std::pair<std::string,float> > name_time_vector;
    int longest_name_length = 0;
    Format best_common_format = MILLISEC;
    for( map_const_iter i = timers.begin(); i != timers.end(); ++i )
    {
      std::string name = i->first;
      float time = i->second.elapsedTime;

      if( (int)name.size() > longest_name_length )
        longest_name_length = name.size();

      if( bestFormat(time) > best_common_format )
        best_common_format = bestFormat(time);

      name_time_vector.push_back(std::pair<std::string,float>(name,time));
    }

    // sort
    if( ascending )
    {
      if( sorting == NAME )
        std::sort(name_time_vector.begin(), name_time_vector.end());
      else
        std::sort(name_time_vector.begin(), name_time_vector.end(), compare_time());
    }
    else
    {
      if( sorting == NAME )
        std::sort(name_time_vector.rbegin(), name_time_vector.rend());
      else
        std::sort(name_time_vector.rbegin(), name_time_vector.rend(), compare_time());
    }


    for( int i = 0; i < (int)name_time_vector.size(); ++i )
    {
      std::string name = name_time_vector[i].first;
      float time = name_time_vector[i].second;
      stream << std::setw(longest_name_length+1) << name;
      if( format == AUTO_COMMON )
        stream << elapsedTimeToString(time, best_common_format);
      else
        stream << elapsedTimeToString(time, format);
      stream << std::endl;
    }
  }

}




void timer::start( const std::string& name )
{
  if( timers.count(name) != 1 )
  {
    Timer& t = timers[name];
    cudaEventCreate( &t.start, 0 );
    cudaEventCreate( &t.stop, 0 );
    t.running = false;
    t.elapsedTime = 0.f;
  }
  if( !timers[name].running )
  {
    timers[name].running = true;
    cudaEventRecord( timers[name].start, 0 );
  }
}


void timer::stop( const std::string& name )
{
  if( timers.count(name) != 1 )
    return;
  if( timers[name].running )
  {
    cudaEventRecord( timers[name].stop, 0 );
    cudaEventSynchronize( timers[name].stop );
    timers[name].running = false;
    float time;
    cudaEventElapsedTime( &time, timers[name].start, timers[name].stop );
    timers[name].elapsedTime += time;
  }
}


void timer::reset( const std::string& name )
{
  if( timers.count(name) != 1 )
    return;
  if( timers[name].running )
  {
    stop(name);
    start(name);
  }
  timers[name].elapsedTime = 0.f;
}


void timer::clear()
{
  std::map<std::string, Timer>::iterator iter = timers.begin();
  std::map<std::string, Timer>::iterator end = timers.end();
  while( iter != end )
  {
    std::string name = iter->first;
    Timer& timer = iter->second;
    if( timer.running )
    {
      stop(name);
    }
    cudaEventDestroy(timer.start);
    cudaEventDestroy(timer.stop);
    ++iter;
  }
  timers.clear();
}


bool timer::isRunning( const std::string& name )
{
  if( timers.count(name) != 1 )
    return false;
  else
    return timers[name].running;
}


float& timer::elapsedTime( const std::string& name )
{
  if( timers.count(name) != 1 )
    return dummy_elapsedTime;
  else
    return timers[name].elapsedTime;
}


void timer::printToScreen( const std::string& name,
                               Format format,
                               Sorting sorting,
                               bool ascending )
{
  if( name == std::string() )
    printTimers( std::cout, timers, format, sorting, ascending );
  else if( timers.count(name) )
  {
    std::map<std::string,Timer> tmp;
    tmp[name] = timers[name];
    printTimers( std::cout, tmp, format, sorting, ascending );
  }
}


void timer::printToFile( const std::string& filename, 
                             const std::string& name,
                             Format format,
                             Sorting sorting,
                             bool ascending )

{
  std::ofstream file( filename.c_str() );

  if( name == std::string() )
    printTimers( file, timers, format, sorting, ascending );
  else if( timers.count(name) )
  {
    std::map<std::string,Timer> tmp;
    tmp[name] = timers[name];
    printTimers( file, tmp, format, sorting, ascending );
  }
}
