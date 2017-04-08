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
	string id;
	  string idBacteria;
	  string idLocation;

	  Entity bacteria;
	  Entity location;

	  bool equals(const Relation& another) {
		  if(bacteria.equals(another.bacteria) && location.equals(another.location))
			  return true;
		  else
			  return false;
	  }

	  void setId(int _id) {
		stringstream ss;
		ss<<"R"<<_id;
		id = ss.str();
	  }

	  void setBacId(const string& bacId) {
		  idBacteria = bacId;
		  bacteria.id = bacId;
	  }

	  void setLocId(const string& locId) {
		  idLocation = locId;
		  location.id = locId;
	  }
};



#endif /* RELATION_H_ */
