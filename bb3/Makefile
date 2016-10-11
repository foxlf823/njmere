cc=g++

#cflags = -O0 -g3 -w \
	-I/home/fox/eclipse/workspace/FoxUtil \
	-msse3 -I /home/fox/Downloads/mshadow-master/mshadow -DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 \
	-I/home/fox/Downloads/LibN3L-master -DUSE_CUDA=0 \
#	-I/home/fox/Downloads/wordnet-blast-master \
	
cflags = -O3 -w \
	-I/home/fox/eclipse/workspace/FoxUtil \
	-msse3 -I /home/fox/Downloads/mshadow-master/mshadow -DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 \
	-I/home/fox/Downloads/LibN3L-master -DUSE_CUDA=0 \
	-static-libgcc -static-libstdc++\

libs = -lm -lopenblas -Wl,-rpath,./ \
	/usr/lib/libwordnet.a \
#	/home/fox/Downloads/wordnet-blast-master/lib/libwnb.a

all: bb3

bb3: bb3.cpp Entity.h Relation.h Document.h Tool.h Options.h utils.h
	$(cc) -o bb3 bb3.cpp $(cflags) $(libs)

	




clean:
	rm -rf *.o
	rm -rf bb3

