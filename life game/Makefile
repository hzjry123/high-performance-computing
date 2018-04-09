all:
	icpc shear.cc -o shear -fopenmp
	icpc shear.cc -o shear_serial -w
	icpc life.cc -o life -fopenmp
	icpc life.cc -o life_serial -w

mac:
	g++ shear.cc -o shear -fopenmp
	g++ life.cc -o life -fopenmp

clean:
	rm -f life life_serial shear shear_serial