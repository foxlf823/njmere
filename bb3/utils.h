/*
 * utils.h
 *
 *  Created on: Dec 20, 2015
 *      Author: fox
 */

#ifndef UTILS_H_
#define UTILS_H_

/*
 * cdr.cpp
 *
 *  Created on: Dec 19, 2015
 *      Author: fox
 */

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
#include "Example.h"

using namespace std;

#define MAX_RELATION 2

void outputResults(const string& docID, const vector<Example>& examples, const string& dir) {
	ofstream m_outf;
	string path = dir+"/BB-event-"+docID+".a2";
	m_outf.open(path.c_str());

	int count=1;
	for(int i=0;i<examples.size();i++) {

		m_outf << "R"<<count << "\t"<< "Lives_In"<<" Bacteria:"<<examples[i].idBacteria<<
			" Location:"<<examples[i].idLocation<<endl;

		count++;
	}
	m_outf.close();
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

/*			if(docs[count].id=="25562320")
				cout<<endl;*/

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
					token.sst = splitted[5];
					sent.tokens.push_back(token);
				}



			}

			ifs.close();
			count++;
		}
	}

}

int parseBB2(const string& dirPath, vector<Document>& documents)
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


			if(string::npos != filePath.find(".a1")) { // entity
				Document doc;
				doc.id = fileName.substr(fileName.find_last_of("-")+1, fileName.find(".")-fileName.find_last_of("-")-1);


				ifstream ifs;
				ifs.open(filePath.c_str());

				string line;
				while(getline(ifs, line)) {

					if(!line.empty()) {
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
						} else { // non-continuous entity
							entity.type = temp1[0];
							entity.begin = atoi(temp1[1].c_str());
							entity.end = atoi(temp1[2].c_str());
							entity.begin2 = atoi(temp1[3].c_str());
							entity.end2 = atoi(temp1[4].c_str());
						}
						doc.entities.push_back(entity);


					}
				}
				ifs.close();
				documents.push_back(doc);
			} else if(string::npos != fileName.find(".a2")) { // relation
				Document& doc = documents[documents.size()-1];

				ifstream ifs;
				ifs.open(filePath.c_str());

				string line;
				while(getline(ifs, line)) {

					if(!line.empty()) {
						vector<string> splitted;
						fox::split(line, splitted, "\t| |:");
						if(splitted[1]=="Localization") {
							Relation relation;

							relation.idBacteria = splitted[3];
							relation.idLocation = splitted[5];

							doc.relations.push_back(relation);
						}

					}
				}
				ifs.close();
			} else if(string::npos != fileName.find(".txt")) { // txt
				Document& doc = documents[documents.size()-1];

				ifstream ifs;
				ifs.open(filePath.c_str());

				string line;
				int count = 1;
				while(getline(ifs, line)) {

					if(!line.empty()) {
						if(count == 1) {
							doc.title = line;
							count++;
						} else {
							if(doc.abstract.empty())
								doc.abstract = line;
							else
								doc.abstract += " "+line;
						}
					}
				}
				ifs.close();
			}




		}
	}

    return 0;

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


			if(string::npos != filePath.find(".a1")) { // entity
				Document doc;
				doc.id = fileName.substr(fileName.find_last_of("-")+1, fileName.find(".")-fileName.find_last_of("-")-1);


				ifstream ifs;
				ifs.open(filePath.c_str());

				string line;
				while(getline(ifs, line)) {

					if(!line.empty()) {
						vector<string> splitted;
						fox::split_bychar(line, splitted, '\t');
						if(splitted[1].substr(0, splitted[1].find(" ")) == ("Title")) {
							doc.title = splitted[2];
						} else if(splitted[1].substr(0, splitted[1].find(" ")) == ("Paragraph")) {
							if(doc.abstract.empty())
								doc.abstract = splitted[2];
							else
								doc.abstract += " "+splitted[2];
						} else {
							Entity entity;
							entity.id = splitted[0];
							entity.text = splitted[2];

							vector<string> temp1;
							fox::split(splitted[1], temp1, " |;");

							if(temp1.size() == 3) {
								entity.type = temp1[0];
								entity.begin = atoi(temp1[1].c_str());
								entity.end = atoi(temp1[2].c_str());
							} else { // non-continuous entity
								entity.type = temp1[0];
								entity.begin = atoi(temp1[1].c_str());
								entity.end = atoi(temp1[2].c_str());
								entity.begin2 = atoi(temp1[3].c_str());
								entity.end2 = atoi(temp1[4].c_str());
							}
							doc.entities.push_back(entity);
						}

					}
				}
				ifs.close();
				documents.push_back(doc);
			} else if(string::npos != fileName.find(".a2")) { // relation
				Document& doc = documents[documents.size()-1];

				ifstream ifs;
				ifs.open(filePath.c_str());

				string line;
				while(getline(ifs, line)) {

					if(!line.empty()) {
						vector<string> splitted;
						fox::split(line, splitted, "\t| |:");
						if(splitted.size() == 6) {
							Relation relation;

							relation.idBacteria = splitted[3];
							relation.idLocation = splitted[5];

							doc.relations.push_back(relation);
						}

					}
				}
				ifs.close();
			} else { // txt

			}




		}
	}


    return 0;

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

bool isTokenInEntity(const fox::Token& tok, const Entity& entity) {

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

bool isLoc(const Entity& bacteria, const Entity& location, const Document& doc) {

	for(int i=0;i<doc.relations.size();i++) {
		if(doc.relations[i].idBacteria == bacteria.id && doc.relations[i].idLocation == location.id)
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






#endif /* UTILS_H_ */
