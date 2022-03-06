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
