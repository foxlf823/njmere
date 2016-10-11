
#ifndef UTILS_H_
#define UTILS_H_


#include <stdio.h>
#include <vector>
#include "Word2Vec.h"
#include "Utf.h"
#include "Entity.h"
#include "Relation.h"
#include "Token.h"
#include "FoxUtil.h"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <list>
#include <sstream>
#include "ADEsentence.h"
#include "Sent.h"

using namespace std;

// schema BILOU, three entity types (Bacteria,Habitat,Geographical)
#define TYPE_Disease "Disease"
#define TYPE_Chemical "Chemical"
#define MAX_ENTITY 9
#define B_Disease "B_Disease"
#define I_Disease "I_Disease"
#define L_Disease "L_Disease"
#define U_Disease "U_Disease"
#define B_Chemical "B_Chemical"
#define I_Chemical "I_Chemical"
#define L_Chemical "L_Chemical"
#define U_Chemical "U_Chemical"
#define OTHER "O"


void appendEntity(const fox::Token& token, Entity& entity) {
	int whitespacetoAdd = token.begin-entity.end;
	for(int j=0;j<whitespacetoAdd;j++)
		entity.text += " ";
	entity.text += token.word;
	entity.end = token.end;
}

void newEntity(const fox::Token& token, const string& labelName, Entity& entity) {
	entity.type = labelName.substr(labelName.find("_")+1);
	entity.begin = token.begin;
	entity.end = token.end;
	entity.text = token.word;
}

int NERlabelName2labelID(const string& labelName) {
	if(labelName == B_Disease) {
		return 0;
	} else if(labelName == I_Disease) {
		return 1;
	} else if(labelName == L_Disease) {
		return 2;
	} else if(labelName == U_Disease) {
		return 3;
	} else if(labelName == B_Chemical) {
		return 4;
	} else if(labelName == I_Chemical) {
		return 5;
	} else if(labelName == L_Chemical) {
		return 6;
	} else if(labelName == U_Chemical) {
		return 7;
	} else
		return 8;
}

string NERlabelID2labelName(const int labelID) {
	if(labelID == 0) {
		return B_Disease;
	} else if(labelID == 1) {
		return I_Disease;
	} else if(labelID == 2) {
		return L_Disease;
	} else if(labelID == 3) {
		return U_Disease;
	} else if(labelID == 4) {
		return B_Chemical;
	} else if(labelID == 5) {
		return I_Chemical;
	} else if(labelID == 6) {
		return L_Chemical;
	} else if(labelID == 7) {
		return U_Chemical;
	} else
		return OTHER;
}

// two relation type (Lives_In or not)
#define MAX_RELATION 2
#define ADE "ADE"
#define Not_ADE "Not_ADE"



int RellabelName2labelID(const string& labelName) {

	if(labelName == ADE) {
		return 0;
	} else
		return 1;

}

string RellabelID2labelName(const int labelID) {

	if(labelID == 0) {
		return ADE;
	} else
		return Not_ADE;

}

void newRelation(Relation& relation, const Entity& entity1, const Entity& entity2) {

	relation.entity1 = entity1;
	relation.entity2 = entity2;
}

void loadProcessedFile(const string& dirPath, vector < vector< fox::Sent > > & groups) {

	struct dirent** namelist = NULL;
	int total = scandir(dirPath.c_str(), &namelist, 0, alphasort);

	for(int i=0;i<total;i++) {

		if (namelist[i]->d_type == 8) {
			//file
			if(namelist[i]->d_name[0]=='.')
				continue;

			string filePath = dirPath;
			filePath += "/";
			filePath += namelist[i]->d_name;
			string fileName = namelist[i]->d_name;

			ifstream ifs;
			ifs.open(filePath.c_str());
			fox::Sent sent;
			string line;
			vector<fox::Sent> group;

			while(getline(ifs, line)) {
				if(line.empty()){
					// new line
					if(sent.tokens.size()!=0) {
						group.push_back(sent);
						sent.tokens.clear();
					}
				} else {
					vector<string> splitted;
					fox::split_bychar(line, splitted, '\t');
					fox::Token token;
					token.word = splitted[0];
					token.begin = atoi(splitted[1].c_str());
					token.end = atoi(splitted[2].c_str());
					token.pos = splitted[3];
					token.lemma = splitted[4];
					token.depGov = atoi(splitted[5].c_str());
					token.depType = splitted[6];
					sent.tokens.push_back(token);
				}


			}

			ifs.close();

			groups.push_back(group);
		}
	}

}


void loadAnnotatedFile(const string& dirPath, vector< vector<ADEsentence> > & groups)
{
	struct dirent** namelist = NULL;
	int total = scandir(dirPath.c_str(), &namelist, 0, alphasort);


	for(int i=0;i<total;i++) {

		if (namelist[i]->d_type == 8) {
			//file
			if(namelist[i]->d_name[0]=='.')
				continue;

			string filePath = dirPath;
			filePath += "/";
			filePath += namelist[i]->d_name;
			string fileName = namelist[i]->d_name;

			vector<ADEsentence> group;
			ifstream ifs;
			ifs.open(filePath.c_str());
			string line;
			ADEsentence sentence;

			while(getline(ifs, line)) {
				if(line.empty()) {
					group.push_back(sentence);
					sentence.clear();
				} else {
					vector<string> splitted;
					fox::split_bychar(line, splitted, '\t');
					if(splitted[0]=="offset") {
						sentence.offset = atoi(splitted[1].c_str());
					} else if(splitted[0]=="EN") {
						Entity entity;
						entity.text = splitted[1];
						entity.type = splitted[2];
						entity.begin = atoi(splitted[3].c_str());
						entity.end = atoi(splitted[4].c_str());
						sentence.entities.push_back(entity);
					} else if(splitted[0]=="ADE") {
						Entity entity1;
						entity1.text = splitted[1];
						entity1.type = TYPE_Disease;
						entity1.begin = atoi(splitted[2].c_str());
						entity1.end = atoi(splitted[3].c_str());
						Entity entity2;
						entity2.text = splitted[4];
						entity2.type = TYPE_Chemical;
						entity2.begin = atoi(splitted[5].c_str());
						entity2.end = atoi(splitted[6].c_str());
						Relation relation;
						relation.entity1 = entity1;
						relation.entity2 = entity2;
						sentence.relations.push_back(relation);
					} else {
						sentence.text = splitted[0];
					}
				}

			}

			ifs.close();

			groups.push_back(group);

		}
	}




}


bool isTokenBeforeEntity(const fox::Token& tok, const Entity& entity) {
	if(tok.begin<entity.begin)
		return true;
	else
		return false;
}

bool isTokenAfterEntity(const fox::Token& tok, const Entity& entity) {
	if(entity.end2 == -1) {
		if(tok.end>entity.end)
			return true;
		else
			return false;

	} else {
		if(tok.end>entity.end2)
			return true;
		else
			return false;

	}

}


string isTokenInEntity(const fox::Token& tok, const Entity& entity) {

	if(tok.begin==entity.begin && tok.end==entity.end)
		return "U";
	else if(tok.begin==entity.begin)
		return "B";
	else if(tok.end==entity.end)
		return "L";
	else if(tok.begin>entity.begin && tok.end<entity.end)
		return "I";
	else
		return "O";

}

bool boolTokenInEntity(const fox::Token& tok, const Entity& entity) {

	if(entity.end2 == -1) {
		if(tok.begin>=entity.begin && tok.end<=entity.end)
			return true;
		else
			return false;
	} else {

		if((tok.begin>=entity.begin && tok.end<=entity.end) ||
				(tok.begin>=entity.begin2 && tok.end<=entity.end2))
			return true;
		else
			return false;

	}
}

bool isTokenBetweenTwoEntities(const fox::Token& tok, const Entity& former, const Entity& latter) {

	if(former.end2 == -1) {
		if(tok.begin>=former.end && tok.end<=latter.begin)
			return true;
		else
			return false;
	} else {
		if(tok.begin>=former.end2 && tok.end<=latter.begin)
			return true;
		else
			return false;
	}
}


void deleteEntity(vector<Entity>& entities, const Entity& target)
{
	vector<Entity>::iterator iter = entities.begin();
	for(;iter!=entities.end();iter++) {
		if((*iter).equals(target)) {
			break;
		}
	}
	if(iter!=entities.end()) {
		entities.erase(iter);
	}
}

int containsEntity(vector<Entity>& source, const Entity& target) {

	for(int i=0;i<source.size();i++) {
		if(source[i].equals(target))
			return i;

	}

	return -1;
}

int containsRelation(vector<Relation>& source, const Relation& target) {

	for(int i=0;i<source.size();i++) {
		if(source[i].equals(target))
			return i;

	}

	return -1;
}

int relationContainsEntity(vector<Relation>& source, const Entity& target) {

	for(int i=0;i<source.size();i++) {
		if(source[i].entity1.equals(target))
			return i;
		else if(source[i].entity2.equals(target))
			return i;

	}

	return -1;
}


bool isADE(const Entity& disease, const Entity& chemical, const ADEsentence & sent) {

	for(int i=0;i<sent.relations.size();i++) {
		if(sent.relations[i].entity1.equals(disease) && sent.relations[i].entity2.equals(chemical))
			return true;
	}

	return false;

}

double precision(int correct, int predict) {
	return 1.0*correct/predict;
}

double recall(int correct, int gold) {
	return 1.0*correct/gold;
}

double f1(int correct, int gold, int predict) {
	double p = precision(correct, predict);
	double r = recall(correct, gold);

	return 2*p*r/(p+r);
}

double f1(double p, double r) {

	return 2*p*r/(p+r);
}




#endif /* UTILS_H_ */
