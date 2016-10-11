

#ifndef ADESENTENCE_H_
#define ADESENTENCE_H_

#include <string>
#include "Entity.h"
#include "Relation.h"

using namespace std;


class ADEsentence {
public:
	ADEsentence() {
		offset = -1;
	}

	void clear() {
		entities.clear();
		relations.clear();
		text.clear();
		offset = -1;
	}

	vector<Entity> entities;
	vector<Relation> relations;
	string text;
	int offset;
};

#endif /* ADESENTENCE_H_ */
