/*
 * beamsearch.h
 *
 *  Created on: Jun 14, 2016
 *      Author: fox
 */

#ifndef BEAMSEARCH_H_
#define BEAMSEARCH_H_
#include <list>
#include <vector>
#include "Prediction.h"

using namespace std;

// insert a prediction into buffer and ordered, the first one with the largest score
void addToBuffer(list<Prediction> & buffer, Prediction& predict) {

	if(buffer.size()==0) {
		buffer.push_back(predict);
		return;
	} else {
		bool inserted = false;
		list<Prediction>::iterator iter;
		for(iter=buffer.begin();iter!=buffer.end();iter++) {
			if(predict.getSum()>(*iter).getSum()) {
				buffer.insert(iter, predict);
				inserted = true;
				break;
			}
		}

		if(!inserted)
			buffer.push_back(predict);
	}
}

void addToBufferByAvg(list<Prediction> & buffer, Prediction& predict) {

	if(buffer.size()==0) {
		buffer.push_back(predict);
		return;
	} else {
		bool inserted = false;
		list<Prediction>::iterator iter;
		for(iter=buffer.begin();iter!=buffer.end();iter++) {
			if(predict.getAvg()>(*iter).getAvg()) {
				buffer.insert(iter, predict);
				inserted = true;
				break;
			}
		}

		if(!inserted)
			buffer.push_back(predict);
	}
}

void copyPrediction(const Prediction & old, Prediction & newOne) {
	for(int i=0;i<old.labels.size();i++) {
		newOne.addScore(old.getScore(i));
		newOne.labels.push_back(old.labels[i]);
	}
}

void Kbest(vector<Prediction> & beam, list<Prediction> & buffer, int beamSize) {
	int count=1;
	list<Prediction>::iterator iter;
	for(iter=buffer.begin(); iter!=buffer.end(); iter++) {
		if(count > beamSize)
			break;
		beam.push_back((*iter));
		count++;
	}

}

bool earlyUpdate(vector<Prediction> & beam, const Prediction & gold) {
	for(int i=0;i<beam.size();i++) {
		if(beam[i].equal(gold))
			return false;
	}

	return true;
}



#endif /* BEAMSEARCH_H_ */
