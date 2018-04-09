/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to High Performance Computing Systems
 * Instructor: Bryan Mills, PhD
 * Code to compute an approximation of pi on MPI.
 * Original source: www.mcs.anl.gov
 */

#include "mpi.h"
#include <math.h>

int main(argc,argv)
     int argc;
     char *argv[];
{
  int done = 0, n, myid, numprocs, i;
  double PI25DT = 3.141592653589793238462643;
  double mypi, pi, h, sum, x;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  
  if (myid == 0) {
    sscanf(argv[1],"%d",&n);
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  h   = 1.0 / (double) n;
  sum = 0.0;
  for (i = myid + 1; i <= n; i += numprocs) {
    x = h * ((double)i - 0.5);
    sum += 4.0 / (1.0 + x*x);
  }
  mypi = h * sum;
    
  MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0,
	     MPI_COMM_WORLD);
    
  if (myid == 0)
    printf("\npi is approximately %.16f, Error is %.16f\n",
	   pi, fabs(pi - PI25DT));
    
  MPI_Finalize();
  return 0;
}
