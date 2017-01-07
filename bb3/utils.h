
#ifndef UTILS_H_
#define UTILS_H_


#include <stdio.h>
#include <vector>
#include "Word2Vec.h"
#include "Utf.h"
#include "Entity.h"
#include "Token.h"
#include "FoxUtil.h"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "Document.h"
#include "NerExample.h"
#include <list>
#include <sstream>

using namespace std;

#define USE_IMP 0

// schema BILOU, three entity types (Bacteria,Habitat,Geographical)
#define TYPE_Bac "Bacteria"
#define TYPE_Hab "Habitat"
#define TYPE_Geo "Geographical"
#define MAX_ENTITY 13
#define B_Bacteria "B_Bacteria"
#define I_Bacteria "I_Bacteria"
#define L_Bacteria "L_Bacteria"
#define U_Bacteria "U_Bacteria"
#define B_Habitat "B_Habitat"
#define I_Habitat "I_Habitat"
#define L_Habitat "L_Habitat"
#define U_Habitat "U_Habitat"
#define B_Geographical "B_Geographical"
#define I_Geographical "I_Geographical"
#define L_Geographical "L_Geographical"
#define U_Geographical "U_Geographical"
#define OTHER "O"


void appendEntity(const fox::Token& token, Entity& entity) {
	int whitespacetoAdd = token.begin-entity.end;
	for(int j=0;j<whitespacetoAdd;j++)
		entity.text += " ";
	entity.text += token.word;
	entity.end = token.end;
}

void newEntity(const fox::Token& token, const string& labelName, Entity& entity, int entityId) {
	stringstream ss;
	ss<<"T"<<entityId;
	entity.id = ss.str();
	entity.type = labelName.substr(labelName.find("_")+1);
	entity.begin = token.begin;
	entity.end = token.end;
	entity.text = token.word;
}

int NERlabelName2labelID(const string& labelName) {
	if(labelName == B_Bacteria) {
		return 0;
	} else if(labelName == I_Bacteria) {
		return 1;
	} else if(labelName == L_Bacteria) {
		return 2;
	} else if(labelName == U_Bacteria) {
		return 3;
	} else if(labelName == B_Habitat) {
		return 4;
	} else if(labelName == I_Habitat) {
		return 5;
	} else if(labelName == L_Habitat) {
		return 6;
	} else if(labelName == U_Habitat) {
		return 7;
	} else if(labelName == B_Geographical) {
		return 8;
	} else if(labelName == I_Geographical)
		return 9;
	else if(labelName == L_Geographical)
			return 10;
	else if(labelName == U_Geographical)
			return 11;
	else
		return 12;
}

string NERlabelID2labelName(const int labelID) {
	if(labelID == 0) {
		return B_Bacteria;
	} else if(labelID == 1) {
		return I_Bacteria;
	} else if(labelID == 2) {
		return L_Bacteria;
	} else if(labelID == 3) {
		return U_Bacteria;
	} else if(labelID == 4) {
		return B_Habitat;
	} else if(labelID == 5) {
		return I_Habitat;
	} else if(labelID == 6) {
		return L_Habitat;
	} else if(labelID == 7) {
		return U_Habitat;
	} else if(labelID == 8) {
		return B_Geographical;
	} else if(labelID == 9)
		return I_Geographical;
	else if(labelID == 10)
			return L_Geographical;
	else if(labelID == 11)
			return U_Geographical;
	else
		return OTHER;
}

// two relation type (Lives_In or not)
#if USE_IMP

	#define MAX_RELATION 4
	#define Lives_In "Lives_In"
	#define Not_Lives_In "Not_Lives_In"
	#define IMP_BAC "IMP_BAC"
	#define IMP_LOC "IMP_LOC"

#else

#define MAX_RELATION 2
#define Lives_In "Lives_In"
#define Not_Lives_In "Not_Lives_In"

#endif

int RellabelName2labelID(const string& labelName) {
#if USE_IMP

	if(labelName == Lives_In) {
		return 0;
	} else if(labelName == IMP_BAC)
		return 1;
	else if(labelName == IMP_LOC)
		return 2;
	else
		return 3;

#else
	if(labelName == Lives_In) {
		return 0;
	} else
		return 1;
#endif
}

string RellabelID2labelName(const int labelID) {
#if USE_IMP

	if(labelID == 0) {
		return Lives_In;
	} else if(labelID == 1)
		return IMP_BAC;
	else if(labelID == 2)
		return IMP_LOC;
	else
		return Not_Lives_In;

#else
	if(labelID == 0) {
		return Lives_In;
	} else
		return Not_Lives_In;
#endif
}

void newRelation(Relation& relation, const Entity& bac, const Entity& loc, int relationId) {
	stringstream ss;
	ss<<"R"<<relationId;
	relation.id = ss.str();
	relation.idBacteria = bac.id;
	relation.idLocation = loc.id;

	relation.bacteria = bac;
	relation.location = loc;
}

void outputResults(const string& id, vector<Entity>& entities, vector<Relation>& relations, const string& dir) {
	ofstream m_outf;
	string path = dir+"/BB-event+ner-"+id+".a2";
	m_outf.open(path.c_str());

	for(int i=0;i<entities.size();i++) {

		m_outf << entities[i].id << "\t"<< entities[i].type<<" "<<entities[i].begin<<" "<<
				entities[i].end<<"\t"<<entities[i].text<<endl;

	}

	for(int i=0;i<relations.size();i++) {
		m_outf << relations[i].id << "\t"<< "Lives_In"<<" Bacteria:"<<relations[i].idBacteria<<
			" Location:"<<relations[i].idLocation<<endl;
	}

	m_outf.close();
}

void outputEnityResults(const string& id, vector<Entity>& entities, const string& dir) {
	ofstream m_outf;
	string path = dir+"/BB-event+ner-"+id+".a2";
	m_outf.open(path.c_str());

	int count=1;
	for(int i=0;i<entities.size();i++) {

		m_outf << entities[i].id << "\t"<< entities[i].type<<" "<<entities[i].begin<<" "<<
				entities[i].end<<"\t"<<entities[i].text<<endl;

		count++;
	}
	m_outf.close();
}

int findEntityById(const string& id, const vector<Entity>& entities) {

	for(int i=0;i<entities.size();i++) {
		if(entities[i].id==id) {
			return i;
		}

	}

	return -1;
}

void loadNlpFile(const string& dirPath, vector<Document>& docs) {

	struct dirent** namelist = NULL;
	int total = scandir(dirPath.c_str(), &namelist, 0, alphasort);
	int count = 0;

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


			while(getline(ifs, line)) {
				if(line.empty()){
					// new line
					if(sent.tokens.size()!=0) {
						docs[count].sents.push_back(sent);
						docs[count].sents[docs[count].sents.size()-1].begin = sent.tokens[0].begin;
						docs[count].sents[docs[count].sents.size()-1].end = sent.tokens[sent.tokens.size()-1].end;
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
			count++;
		}
	}

}

bool isEntityOverlapped(const Entity& former, const Entity& latter) {
	if(former.end2==-1) {
		if(former.end<=latter.begin)
			return false;
		else
			return true;
	} else {
		if(former.end2<=latter.begin)
			return false;
		else
			return true;
	}
}


int parseBB3(const string& dirPath, vector<Document>& documents)
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

			if(string::npos != filePath.find(".a1")) { // doc
				Document doc;
				doc.id = fileName.substr(fileName.find_last_of("-")+1, fileName.find(".")-fileName.find_last_of("-")-1);
				doc.maxParagraphId = -1;
				ifstream ifs;
				ifs.open(filePath.c_str());
				string line;
				while(getline(ifs, line)) {

					if(!line.empty() && line[0]=='T') {
						if(doc.maxParagraphId < atoi(line.substr(1,1).c_str()))
							doc.maxParagraphId = atoi(line.substr(1,1).c_str());
					}
				}

				ifs.close();

				documents.push_back(doc);
			} else if(string::npos != filePath.find(".a2")) { // entity && relation
				Document& doc = documents[documents.size()-1];

				ifstream ifs;
				ifs.open(filePath.c_str());


				string line;
				while(getline(ifs, line)) {

					if(!line.empty()) {

						if(line[0] == 'T') { // entity
							vector<string> splitted;
							fox::split_bychar(line, splitted, '\t');
							Entity entity;
							entity.id = splitted[0];
							entity.text = splitted[2];

							vector<string> temp1;
							fox::split(splitted[1], temp1, " |;");

							if(temp1.size() == 3) {
								entity.type = temp1[0];
								entity.begin = atoi(temp1[1].c_str());
								entity.end = atoi(temp1[2].c_str());

								if(doc.entities.size()>0) {
									Entity& former = doc.entities[doc.entities.size()-1];
									if(isEntityOverlapped(former, entity)) {
										// if two entities is overlapped, we keep the one with narrow range
										int formerRange = former.end-former.begin;
										int entityRange = entity.end-entity.begin;
										if(entityRange<formerRange) {
											doc.entities.pop_back();
											doc.entities.push_back(entity);
										}
									} else
										doc.entities.push_back(entity);
								} else
									doc.entities.push_back(entity);

/*								cout<<"dump entity#####"<<endl;
								for(int i=0;i<doc.entities.size();i++)
									cout<<doc.entities[i].id<<endl;*/

							} else { // non-continuous entity is ignored

/*								entity.type = temp1[0];
								entity.begin = atoi(temp1[1].c_str());
								entity.end = atoi(temp1[2].c_str());
								entity.begin2 = atoi(temp1[3].c_str());
								entity.end2 = atoi(temp1[4].c_str());
								doc.entities.push_back(entity);*/
							}

						} else if(line[0] == 'R') { //relation
							vector<string> splitted;
							fox::split(line, splitted, "\t| |:");
							Relation relation;

							relation.idBacteria = splitted[3];
							relation.idLocation = splitted[5];


							int idxBac = findEntityById(relation.idBacteria, doc.entities);
							int idxLoc = findEntityById(relation.idLocation, doc.entities);
							if(idxBac ==-1 || idxLoc == -1)
								continue; // we get rid of some entities

							relation.bacteria = doc.entities[idxBac];
							relation.location = doc.entities[idxLoc];

							doc.relations.push_back(relation);
						}

					}
				}
				ifs.close();

			}

		}
	}


    return 0;

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

// delete the entities not in the window sentence
void deleteEntityOutOfWindow(vector<Entity>& entities, int begin, int end) {


	vector<Entity>::iterator iter = entities.begin();
	for(;iter!=entities.end();) {
		if((*iter).begin < begin || (*iter).end > end) {
			iter = entities.erase(iter);
		} else {
			iter++;
		}
	}

}

void findEntityInWindow(int begin, int end, vector<Entity>& src, vector<Entity>& dst) {
	for(int i=0;i<src.size();i++) {
		if(src[i].begin2==-1) {
			if(src[i].begin >= begin && src[i].end <= end)
				dst.push_back(src[i]);
		} else {
			if(src[i].begin >= begin && src[i].end2 <= end)
				dst.push_back(src[i]);
		}

	}

	return;
}

// sentence spans from begin(include) to end(exclude), sorted because doc.entities are sorted
void findEntityInSent(int begin, int end, const Document& doc, vector<Entity>& results) {

	for(int i=0;i<doc.entities.size();i++) {
		if(doc.entities[i].begin2==-1) {
			if(doc.entities[i].begin >= begin && doc.entities[i].end <= end)
				results.push_back(doc.entities[i]);
		} else {
			if(doc.entities[i].begin >= begin && doc.entities[i].end2 <= end)
				results.push_back(doc.entities[i]);
		}

	}

	return;
}

void findEntityInSent(int begin, int end, const vector<Entity>& source, vector<Entity>& results) {

	for(int i=0;i<source.size();i++) {
		if(source[i].begin2==-1) {
			if(source[i].begin >= begin && source[i].end <= end)
				results.push_back(source[i]);
		} else {
			if(source[i].begin >= begin && source[i].end2 <= end)
				results.push_back(source[i]);
		}

	}

	return;
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
		if(source[i].bacteria.equals(target))
			return i;
		else if(source[i].location.equals(target))
			return i;

	}

	return -1;
}


bool isLoc(const Entity& bacteria, const Entity& location, const Document& doc) {

/*	for(int i=0;i<doc.relations.size();i++) {
		if(doc.relations[i].idBacteria == bacteria.id && doc.relations[i].idLocation == location.id)
			return true;
	}*/
	for(int i=0;i<doc.relations.size();i++) {
		if(doc.relations[i].bacteria.equals(bacteria) && doc.relations[i].location.equals(location))
				return true;
	}

	return false;

}

bool isLoc(const string& idbacteria, const string& idlocation, const Document& doc) {

	for(int i=0;i<doc.relations.size();i++) {
		if(doc.relations[i].idBacteria == idbacteria && doc.relations[i].idLocation == idlocation)
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




#endif /* UTILS_H_ */
