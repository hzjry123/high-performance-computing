/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems Spring 2017
 * Instructor Bryan Mills, PhD
 * Timing operations.
 */

#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

long startTime;

// Starts timer and resets the elapsed time
long timerStart() {
  sleep(1); // Hack, just to make sure we have 1 second between calling start/stop.
  struct timeval tod;
  gettimeofday(&tod, NULL);
  startTime = tod.tv_sec + (tod.tv_usec * 1.0e-6);
  return startTime;
}

// Stops the timer and returns the elapsed time
long timerStop() {
  struct timeval tod;
  gettimeofday(&tod, NULL);
  return ((tod.tv_sec + (tod.tv_usec * 1.0e-6)) - startTime) * 1000;
}
