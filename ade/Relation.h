/*
 * RelationEntity.h
 *
 *  Created on: Mar 9, 2016
 *      Author: fox
 */

#ifndef RELATION_H_
#define RELATION_H_

#include <string>
#include <sstream>

using namespace std;

// non-directional live-in relation in mention level
class Relation {

public:

	  Entity entity1;
	  Entity entity2;

	  bool equals(const Relation& another) const {
		  if(entity1.equals(another.entity1) && entity2.equals(another.entity2))
			  return true;
		  else
			  return false;
	  }
};



#endif /* RELATION_H_ */
