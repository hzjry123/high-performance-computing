all: mpi_trap mpi_pi

mpi_trap: mpi_trap.c
	mpicc -o mpi_trap mpi_trap.c

mpi_trap: mpi_pi.c
	mpicc -o mpi_pi mpi_pi.c

#clean:
#	rm -f mpi_trap
