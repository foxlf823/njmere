/*
 * BiocDocument.h
 *
 *  Created on: Dec 19, 2015
 *      Author: fox
 */

#ifndef DOCUMENT_H_
#define DOCUMENT_H_

#include <string>
#include "Entity.h"
#include "Relation.h"
#include "Sent.h"

using namespace std;


class Document {
public:
	Document() {

	}
/*	virtual ~BiocDocument() {

	}*/

	string id;
	vector<Entity> entities;
	vector<Relation> relations;
	vector<fox::Sent> sents;
	int maxParagraphId;
};

#endif /* DOCUMENT_H_ */
