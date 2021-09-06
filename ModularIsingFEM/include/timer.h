// timer.h

#include <chrono>

class Timer {

public:
  Timer();
  void Start();
  void Stop();
  double Duration();

  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point stop_time;
};

Timer::Timer() {
  Start();
}

void Timer::Start() {
  start_time = std::chrono::high_resolution_clock::now();
}

void Timer::Stop() {
  stop_time = std::chrono::high_resolution_clock::now();
}

double Timer::Duration() {
  std::chrono::duration<double> dur = stop_time - start_time;
  return dur.count();
}
