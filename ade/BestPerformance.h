/*
 * BestPerformance.h
 *
 *  Created on: Sep 16, 2016
 *      Author: fox
 */

#ifndef BESTPERFORMANCE_H_
#define BESTPERFORMANCE_H_

class BestPerformance {
public:
	BestPerformance() {

	}
	double dev_pEntity = 0;
	double dev_rEntity = 0;
	double dev_f1Entity = 0;
	double dev_pRelation = 0;
	double dev_rRelation = 0;
	double dev_f1Relation = 0;

	double test_pEntity = 0;
	double test_rEntity = 0;
	double test_f1Entity = 0;
	double test_pRelation = 0;
	double test_rRelation = 0;
	double test_f1Relation = 0;
};

#endif /* BESTPERFORMANCE_H_ */
