/*
 * Prediction.h
 *
 *  Created on: Jun 13, 2016
 *      Author: fox
 */

#ifndef PREDICTION_H_
#define PREDICTION_H_
#include <vector>
#include <string>
#include "Example.h"
#include "Entity.h"
#include "Relation.h"
#include <cmath>

using namespace std;

// denotes a sentence
class Prediction {
public:
	Prediction(bool g) {
		isGold = g;
		sum = 0;
		expsum = 0;
	}

	bool isGold;

	// both
	vector<int> labels;
	vector<Entity> entities;
	vector<Relation> relations;
	vector<string> labelNames;

	// only used by gold
	vector<Example> examples;

	// only used by predicted
	vector<double> scores;
	double sum;
	double expsum;

public:
	void addScore(const double score) {
		scores.push_back(score);
		sum += score;
	}

	void addLogProb(const double prob) {
		double lp = log(prob);
		scores.push_back(lp);
		sum += lp;
	}

	double getScore(int idx) const {
		return scores[idx];
	}

	double getSum() {
		return sum;
	}

	double getAvg() {
		return sum/scores.size();
	}

	bool equal(const Prediction & other) {
		int labelMinSize = other.labels.size()>labels.size() ? labels.size():other.labels.size();
		for(int i=0; i<labelMinSize; i++) {
			if(labels[i] != other.labels[i])
				return false;
		}

		return true;
	}
};




#endif /* PREDICTION_H_ */
