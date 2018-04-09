/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD
 * Timing operations.
 */

#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

struct timeval start;

// Starts timer and resets the elapsed time
void timerStart(){
  gettimeofday(&start, NULL);
}

// Stops the timer and returns the elapsed time
long timerStop(){
  struct timeval end;
  gettimeofday(&end, NULL);
  return 1e6 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
}
