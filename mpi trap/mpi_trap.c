/**
 * University of Pittsburgh
 * Department of Computer Science
 * CS1645: Introduction to HPC Systems
 * Instructor Bryan Mills, PhD (bmills@cs.pitt.edu)
 * STUDENTS: Implement MPI Send/Recv to distribute the calculation of
 * the trapezoidal estimate.
 */

#include <stdio.h>

#include <mpi.h>

/* Calculate local integral  */
double Trap(double left_endpt, double right_endpt, int trap_count, 
   double base_len);    

/* Function we're integrating */
double f(double x); 

int main(void) {
   int my_rank, comm_sz, n = 1024, local_n;   
   double a = 0.0, b = 3.0, h, local_a, local_b;
   double local_int, total_int;
   int source,proc; 

   /* Let the system do what it needs to start up MPI */
   MPI_Init(NULL, NULL);

   /* Get my process rank */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   /* Find out how many processes are being used */
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

   h = (b-a)/n;          /* h is the same for all processes */
   //local_n = n/comm_sz;  /* So is the number of trapezoids  */
   //if(my_rank < n % comm_sz)local_n = n / comm_sz + 1;
   //else local_n =  n / comm_sz 

   /* STUDENTS: part (e) asks you to modify local_n, local_a, and
      local_b so you can handle the case when n is not evenly
      divisable by comm_sz */

   /* Length of each process' interval of
    * integration = local_n*h.  So my interval
    * starts at: */

   /*
   local_a = a + my_rank*local_n*h;
   local_b = local_a + local_n*h;
   local_int = Trap(local_a, local_b, local_n, h);
   */

   if(my_rank < n % comm_sz){
      local_n = n / comm_sz + 1;
      local_a = a + my_rank*local_n*h;
   }
   else{
      local_n = n / comm_sz;
      local_a = a + my_rank*local_n*h + n % comm_sz * h;
   }
   local_b = local_a + local_n*h;
   local_int = Trap(local_a, local_b, local_n, h);

   /* STUDENTS: Part (a-c) Asks you to implement code such that rank 0
   /*           gathers all the local_int's and sum them. Including
      the local_int from rank 0. Do that work here <-> */
   
   
   if(my_rank != 0){
      //greeting,strlen(greeting)+1,MPI_INT,0,0,MPI_COMM_WORLD
      MPI_Send(&local_int,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
   }
   

   /* Hint: For this homework just use Send/Recv */

   /* Print the result */
   if (my_rank == 0) {
      
      total_int = local_int;

      for(proc = 1; proc < comm_sz; proc++){
         MPI_Recv(&local_int,1,MPI_DOUBLE,proc,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
         total_int += local_int;
      }
      
      printf("With n = %d trapezoids, our estimate\n", n);
      printf("of the integral from %f to %f = %.15e\n",
          a, b, total_int);
   }

   /* Shut down MPI */
   MPI_Finalize();

   return 0;
} /*  main  */


/*------------------------------------------------------------------
 * Function:     Trap
 * Purpose:      Serial function for estimating a definite integral 
 *               using the trapezoidal rule
 * Input args:   left_endpt
 *               right_endpt
 *               trap_count 
 *               base_len
 * Return val:   Trapezoidal rule estimate of integral from
 *               left_endpt to right_endpt using trap_count
 *               trapezoids
 */
double Trap(
      double left_endpt  /* in */, 
      double right_endpt /* in */, 
      int    trap_count  /* in */, 
      double base_len    /* in */) {
   double estimate, x; 
   int i;

   estimate = (f(left_endpt) + f(right_endpt))/2.0;
   for (i = 1; i <= trap_count-1; i++) {
      x = left_endpt + i*base_len;
      estimate += f(x);
   }
   estimate = estimate*base_len;

   return estimate;
} /*  Trap  */


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
double f(double x) {
   return x*x;
} /* f */
