
#ifndef SRC_FEATURE_H_
#define SRC_FEATURE_H_

#include <vector>

using namespace std;
class Feature {

public:
	vector<int> words;
	int randomWord;
	vector<int> characters;
	int pos;
	int sst;
	int ner;

public:
	Feature() {
	}
	virtual ~Feature() {

	}

	void clear() {
		words.clear();

	}
};

#endif /* SRC_FEATURE_H_ */
