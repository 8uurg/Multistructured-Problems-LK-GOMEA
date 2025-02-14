//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#include "time.hpp"

long long getCurrentTimeStampInMilliSeconds()
{
  long long result;

  auto result_as_duration = std::chrono::system_clock::now().time_since_epoch();
  result = std::chrono::duration_cast<std::chrono::milliseconds>(result_as_duration).count();

  return result;
}

long long getMilliSecondsRunningSinceTimeStamp (long long startTimestamp)
{
  long long timestamp_now, difference;

  timestamp_now = getCurrentTimeStampInMilliSeconds();

  difference = timestamp_now-startTimestamp;

  return difference;
}
