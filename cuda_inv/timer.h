#ifndef TIMER_H_
#define TIMER_H_

#include <string>

namespace timer
{
  enum Format {MILLISEC, SEC, MIN, HOUR, HHMMSS, AUTO, AUTO_COMMON};
  enum Sorting {NAME, ELAPSED_TIME};
  /*!
   *  Starts the timer with the given name. If the name was not used before
   *  then the timer starts with 0. If the timer is already running then this
   *  statement has no effect.
   */
  void start( const std::string& name = "default" );

  /*!
   *  Stops the timer with the given name. 
   *  Statement has no effect if the timer does not exist
   */
  void stop( const std::string& name = "default" );

  /*!
   *  Resets the timer 'name' to zero.
   *  Statement has no effect if the timer does not exist or is already stopped
   */
  void reset( const std::string& name = "default" );

  /*!
   *  This clears all timers. All running timers will be stopped and deleted.
   */ 
  void clear();

  /*!
   *  Returns whether the timer is running or not.
   *  Returns false if the timer does not exist.
   */
  bool isRunning( const std::string& name = "default" );

  /*!
   *  Returns the elapsed time in milliseconds.
   *  Statement has no effect if the timer does not exist
   */
  float& elapsedTime( const std::string& name = "default" );

  /*!
   *  Prints the elapsed time and the name of the timer on the screen.
   *  If the timer name is empty then all timers are printed
   *  \param name       Name of the timer to print. If empty all known timers 
   *                    are printed
   *  \param format     The format of the printed time. Possible values are:
   *                    MILLISEC use milliseconds as unit
   *                    SEC use seconds as unit
   *                    MIN use minutes as unit
   *                    HOUR use hours as unit
   *                    HHMMSS use 'hh:mm:ss' format
   *                    AUTO use the best unit for each timer
   *                    AUTO_COMMON use the best common unit for all timers
   *  \param sorting    Can be either NAME or ELAPSED_TIME 
   *  \param ascending  If true use ascending order for timer name or elapsed time
   */
  void printToScreen( const std::string& name = std::string(), 
                      Format format=AUTO_COMMON, Sorting sorting=NAME, bool ascending=true );

  /*!
   *  Same as printToScreen() but redirects the output to a file.
   */
  void printToFile( const std::string& filename, 
                    const std::string& name = std::string(),
                    Format format=AUTO_COMMON, Sorting sorting=ELAPSED_TIME, bool ascending=true );

}


#endif /* TIMER_H_ */
