/*
 * N3Lhelper.h
 *
 *  Created on: Dec 27, 2015
 *      Author: fox
 */

#ifndef N3LHELPER_H_
#define N3LHELPER_H_

#include "N3L.h"

using namespace std;


void alphabet2vectormap(const Alphabet& alphabet, vector<string>& vector, map<string, int>& IDs) {

	for (int j = 0; j < alphabet.size(); ++j) {
		string str = alphabet.from_id(j);
		vector.push_back(str);
		IDs.insert(map<string, int>::value_type(str, j));
	}

}

template<typename T>
void array2NRMat(T * array, int sizeX, int sizeY, NRMat<T>& mat) {
	for(int i=0;i<sizeX;i++) {
		for(int j=0;j<sizeY;j++) {
			mat[i][j] = *(array+i*sizeY+j);
		}
	}
}



#endif /* N3LHELPER_H_ */
