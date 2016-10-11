/*
 * Trigger.h
 *
 *  Created on: Mar 25, 2016
 *      Author: fox
 */

#ifndef TRIGGER_H_
#define TRIGGER_H_
#include <set>
#include <string>

using namespace std;

class Trigger {
public:
	Trigger() {

		triggerVerb.insert("detection");
		triggerVerb.insert("detect");
		triggerVerb.insert("obtain");
		triggerVerb.insert("isolate");
		triggerVerb.insert("isolation");
		triggerVerb.insert("strain");
		triggerVerb.insert("grow");
		triggerVerb.insert("growth");
		triggerVerb.insert("culture");
		triggerVerb.insert("carriage");
		triggerVerb.insert("adhesion");
		triggerVerb.insert("find");
		triggerVerb.insert("pathogen");
		triggerVerb.insert("infection");
		triggerVerb.insert("infect");
		triggerVerb.insert("susceptible");
		triggerVerb.insert("colonize");
		triggerVerb.insert("colonization");
		triggerVerb.insert("present");


		triggerPrep.insert("in");
		triggerPrep.insert("from");
		triggerPrep.insert("of");
		triggerPrep.insert("by");
		triggerPrep.insert("on");
		triggerPrep.insert("to");


	}


	string findVerb(const string& lemma) {
		set<string>::iterator it = triggerVerb.find(lemma);
		if(it != triggerVerb.end())
			return *it;
		else
			return "";
	}


	string findPrep(const string& lemma) {
		set<string>::iterator it = triggerPrep.find(lemma);
		if(it != triggerPrep.end())
			return *it;
		else
			return "";
	}
private:
	set<string> triggerVerb;
	set<string> triggerPrep;
};



#endif /* TRIGGER_H_ */
