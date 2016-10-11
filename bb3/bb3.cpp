/*
 * cdr.cpp
 *
 *  Created on: Dec 19, 2015
 *      Author: fox
 */

#include <vector>
#include "utils.h"
#include "FoxUtil.h"
#include <iostream>
#include "Token.h"
#include "SentSplitter.h"
#include "Tokenizer.h"
#include "N3L.h"
#include "Argument_helper.h"
#include "Options.h"
#include "WordNet.h"
#include "Tool.h"
#include "NNbb_rnn.h"

using namespace std;


int main(int argc, char **argv)
{
#if USE_CUDA==1
  InitTensorEngine();
#else
  InitTensorEngine<cpu>();
#endif


	string optionFile;
	string trainFile;
	string devFile;
	string testFile;
	string outputFile;
	string otherDir; // use other to supplement training data
	string trainNlpFile;
	string devNlpFile;
	string testNlpFile;
	string otherNlpDir;
	bool usedev = false;



	dsr::Argument_helper ah;
	ah.new_named_string("train", "", "", "", trainFile);
	ah.new_named_string("dev", "", "", "", devFile);
	ah.new_named_string("test", "", "", "", testFile);
	ah.new_named_string("other", "", "", "", otherDir);
	ah.new_named_string("option", "", "", "", optionFile);
	ah.new_named_string("output", "", "", "", outputFile);
	ah.new_named_string("trainnlp", "", "", "", trainNlpFile);
	ah.new_named_string("devnlp", "", "", "", devNlpFile);
	ah.new_named_string("testnlp", "", "", "", testNlpFile);
	ah.new_named_string("othernlp", "", "", "", otherNlpDir);
	ah.new_flag("usedev", "", "", usedev);
	ah.process(argc, argv);
	cout<<"train file: " <<trainFile <<endl;
	cout<<"dev file: "<<devFile<<endl;
	cout<<"test file: "<<testFile<<endl;
	cout<<"other Dir: "<<otherDir<<endl;
	cout<<"trainnlp file: "<<trainNlpFile<<endl;
	cout<<"devnlp file: "<<devNlpFile<<endl;
	cout<<"testnlp file: "<<testNlpFile<<endl;
	cout<<"othernlp file: "<<otherNlpDir<<endl;
	cout<<"usedev: "<<usedev<<endl;

	Options options;
	options.load(optionFile);

	if(!outputFile.empty())
		options.output = outputFile;

	options.showOptions();

/*	if((options.channelMode & 2) == 2) {
		if(wninit()) {
			cout<<"warning: can't init wordnet"<<endl;
			exit(0);
		}
	}*/

	Tool tool(options);

	NNbb3_rnn nnbb3(options);
	//NNbb3 nnbb3(options);

	nnbb3.train(usedev, trainFile, devFile, testFile, otherDir, tool,
			trainNlpFile, devNlpFile, testNlpFile, otherNlpDir);



#if USE_CUDA==1
  ShutdownTensorEngine();
#else
  ShutdownTensorEngine<cpu>();
#endif

    return 0;

}

