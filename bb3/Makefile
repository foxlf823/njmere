cc=g++

#cflags = -O0 -g3 -w \
	-I/home/fox/project/FoxUtil \
	-msse3 -I /home/fox/Downloads/mshadow-master/mshadow -DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 \
	-I/home/fox/Downloads/LibN3L-master -DUSE_CUDA=0 \
	
cflags = -O3 -w \
	-I/home/fox/project/FoxUtil \
	-msse3 -I /home/fox/Downloads/mshadow-master/mshadow -DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 \
	-I/home/fox/Downloads/LibN3L-master -DUSE_CUDA=0 \
	-static-libgcc -static-libstdc++\

libs = -lm -lopenblas -Wl,-rpath,./ \
 
all: bb3

bb3: bb3.cpp Entity.h Relation.h Document.h Tool.h Options.h utils.h NNbb3.h 
	$(cc) -o bb3 bb3.cpp $(cflags) $(libs)

	




clean:
	rm -rf *.o
	rm -rf bb3

