all: prefix_sum

prefix_sum: prefix_sum.cu
	nvcc -o prefix_sum prefix_sum.cu --gpu-architecture=sm_30

clean:
	rm -f prefix_sum

