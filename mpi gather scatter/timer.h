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

long startTime;

// Starts timer and resets the elapsed time
void timerStart(){
  struct timeval tod;
  gettimeofday(&tod, NULL);
  startTime = tod.tv_sec + (tod.tv_usec * 1.0e-6);
}

// Stops the timer and returns the elapsed time
long timerStop(){
  struct timeval tod;
  gettimeofday(&tod, NULL);
  return ((tod.tv_sec + (tod.tv_usec * 1.0e-6)) - startTime) * 1000;
}
