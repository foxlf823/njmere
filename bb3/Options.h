#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3L.h"

using namespace std;

class Options {
public:

  int wordCutOff;

  dtype initRange;
  int maxIter;
  int batchSize;
  dtype adaEps;
  dtype adaAlpha;
  dtype regParameter;
  dtype dropProb;

  int evalPerIter;

  bool wordEmbFineTune;

  string abbrPath;
  string puncPath;

  int wordcontext;
  int wordEmbSize;
  int otherEmbSize;
  int hiddenSize;
  int rnnHiddenSize;

  int sent_window;

  int verboseIter;

  string output;
  string embFile;

  int beamSize1;
  int beamSize2;

  int poolType;

  Options() {
    wordCutOff = 0;
    initRange = 0.01;
    maxIter = 1000;
    batchSize = 1;
    adaEps = 1e-6;
    adaAlpha = 0.01;
    regParameter = 1e-8;
    dropProb = 0.5;

    wordcontext = 0;
    wordEmbSize = 50;
    otherEmbSize = 50;
    hiddenSize = 150;
    rnnHiddenSize = 100;

    evalPerIter = 1;
    wordEmbFineTune = true;

    abbrPath = "";
    puncPath = "";

    sent_window = 1;
    verboseIter = 0;

    output = "";
    embFile = "";

    beamSize1 = 1;
    beamSize2 = 1;

    poolType = 0;
  }

  Options(const Options& options) {
	  wordCutOff = options.wordCutOff;
	  initRange = options.initRange;
	  maxIter = options.maxIter;
	  batchSize = options.batchSize;
	  adaEps = options.adaEps;
	  adaAlpha = options.adaAlpha;
	  regParameter = options.regParameter;
	  dropProb = options.dropProb;

	  wordcontext = options.wordcontext;
	  wordEmbSize = options.wordEmbSize;
	  otherEmbSize = options.otherEmbSize;
		hiddenSize = options.hiddenSize;
		rnnHiddenSize = options.rnnHiddenSize;

	  evalPerIter = options.evalPerIter;
	  wordEmbFineTune = options.wordEmbFineTune;

		abbrPath = options.abbrPath;
		puncPath = options.puncPath;

		sent_window  = options.sent_window;

		verboseIter = options.verboseIter;

		output = options.output;
		embFile = options.embFile;

		beamSize1 = options.beamSize1;
		beamSize2 = options.beamSize2;

		poolType = options.poolType;
  }

/*  virtual ~Options() {

  }*/

  void setOptions(const vector<string> &vecOption) {
    int i = 0;
    for (; i < vecOption.size(); ++i) {
      pair<string, string> pr;
      string2pair(vecOption[i], pr, '=');
      if (pr.first == "wordCutOff")
        wordCutOff = atoi(pr.second.c_str());
      else if (pr.first == "initRange")
        initRange = atof(pr.second.c_str());
      else if (pr.first == "maxIter")
        maxIter = atoi(pr.second.c_str());
      else if (pr.first == "batchSize")
        batchSize = atoi(pr.second.c_str());
      else if (pr.first == "adaEps")
        adaEps = atof(pr.second.c_str());
      else if (pr.first == "adaAlpha")
        adaAlpha = atof(pr.second.c_str());
      else if (pr.first == "regParameter")
        regParameter = atof(pr.second.c_str());
      else if (pr.first == "dropProb")
        dropProb = atof(pr.second.c_str());

      else if (pr.first == "hiddenSize")
        hiddenSize = atoi(pr.second.c_str());
      else if (pr.first == "rnnHiddenSize")
    	  rnnHiddenSize = atoi(pr.second.c_str());

      else if (pr.first == "wordcontext")
    	  wordcontext = atoi(pr.second.c_str());
      else if (pr.first == "wordEmbSize")
        wordEmbSize = atoi(pr.second.c_str());
      else if(pr.first == "otherEmbSize")
    	  otherEmbSize = atoi(pr.second.c_str());
        

      else if(pr.first == "evalPerIter")
    	  evalPerIter = atoi(pr.second.c_str());

      else if (pr.first == "wordEmbFineTune")
    	  wordEmbFineTune = (pr.second == "true") ? true : false;

      else if(pr.first == "abbrPath")
    	  abbrPath = pr.second;

      else if(pr.first == "puncPath")
    	  puncPath = pr.second;
      else if (pr.first == "sent_window")
    	  sent_window = atoi(pr.second.c_str());
      else if(pr.first == "verboseIter")
    	  verboseIter = atoi(pr.second.c_str());

      else if(pr.first == "output")
          	  output = pr.second;
      else if(pr.first == "embFile")
          embFile = pr.second;

      else if(pr.first == "beamSize1")
    	  beamSize1 = atoi(pr.second.c_str());
      else if(pr.first == "beamSize2")
    	  beamSize2 = atoi(pr.second.c_str());

      else if(pr.first == "poolType")
    	  poolType = atoi(pr.second.c_str());
    }
  }

  void showOptions() {
    std::cout << "wordCutOff = " << wordCutOff << std::endl;
    std::cout << "initRange = " << initRange << std::endl;
    std::cout << "maxIter = " << maxIter << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "adaEps = " << adaEps << std::endl;
    std::cout << "adaAlpha = " << adaAlpha << std::endl;
    std::cout << "regParameter = " << regParameter << std::endl;
    std::cout << "dropProb = " << dropProb << std::endl;

    std::cout << "hiddenSize = " << hiddenSize << std::endl;
    std::cout << "rnnHiddenSize = " << rnnHiddenSize << std::endl;

    std::cout<<"wordcontext = " << wordcontext << endl;
    std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
    std::cout<<"otherEmbSize = "<<otherEmbSize << std::endl;

    cout<< "evalPerIter = " << evalPerIter << endl;
    cout<< "wordEmbFineTune = "<<wordEmbFineTune<<endl;

    cout<<"abbrPath = "<<abbrPath<<endl;
    cout<<"puncPath = "<<puncPath<<endl;

    cout<<"sent_window = "<<sent_window<<endl;
    cout<<"verboseIter = "<<verboseIter<<endl;
    cout<<"output = "<<output<<endl;
    cout<<"embFile = "<<embFile<<endl;

    cout<<"beamSize1 = "<<beamSize1<<endl;
    cout<<"beamSize2 = "<<beamSize2<<endl;

    cout<<"poolType = "<<poolType<<endl;
  }

  void load(const std::string& infile) {
    ifstream inf;
    inf.open(infile.c_str());
    vector<string> vecLine;
    while (1) {
      string strLine;
      if (!my_getline(inf, strLine)) {
        break;
      }
      if (strLine.empty())
        continue;
      vecLine.push_back(strLine);
    }
    inf.close();
    setOptions(vecLine);
  }
};

#endif

