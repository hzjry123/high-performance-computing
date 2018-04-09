all: particles particles_serial

particles: particles.c
	mpicc -o particles particles.c

particles_serial: particles_serial.c
	icpc -o particles_serial particles_serial.c

clean:
	rm -f particles
	rm -f particles_serial
