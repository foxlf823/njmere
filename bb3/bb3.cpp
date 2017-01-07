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
#include "Tool.h"

#if USE_IMP

#include "NNbb_imp.h"
#include "NNbb_1beam_early.h"

#else

#include "NNbb_ner.h"
#include "NNbb.h"
#include "NNbb_multisent.h"
#include "NNbb_coref.h"
#include "NNbb2.h"
#include "NNbb3.h"
#include "NNbb3_error.h"

#endif

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
	string trainNlpFile;
	string devNlpFile;
	string testNlpFile;



	dsr::Argument_helper ah;
	ah.new_named_string("train", "", "", "", trainFile);
	ah.new_named_string("dev", "", "", "", devFile);
	ah.new_named_string("test", "", "", "", testFile);
	ah.new_named_string("option", "", "", "", optionFile);
	ah.new_named_string("output", "", "", "", outputFile);
	ah.new_named_string("trainnlp", "", "", "", trainNlpFile);
	ah.new_named_string("devnlp", "", "", "", devNlpFile);
	ah.new_named_string("testnlp", "", "", "", testNlpFile);


	ah.process(argc, argv);
	cout<<"train file: " <<trainFile <<endl;
	cout<<"dev file: "<<devFile<<endl;
	cout<<"test file: "<<testFile<<endl;

	cout<<"trainnlp file: "<<trainNlpFile<<endl;
	cout<<"devnlp file: "<<devNlpFile<<endl;
	cout<<"testnlp file: "<<testNlpFile<<endl;


	Options options;
	options.load(optionFile);

	if(!outputFile.empty())
		options.output = outputFile;

	options.showOptions();

	Tool tool(options);


	//NNbb_1beam_early nnbb(options);
	//NNbb_multisent nnbb(options);
	//NNbb_coref nnbb(options);
	//NNbb_imp nnbb(options);
	//NNbb nnbb(options);
	//NNbb2 nnbb(options);
	//NNbb3 nnbb(options);
	NNbb3_error nnbb(options);

	nnbb.trainAndTest(trainFile, devFile, testFile, tool,
			trainNlpFile, devNlpFile, testNlpFile);



#if USE_CUDA==1
  ShutdownTensorEngine();
#else
  ShutdownTensorEngine<cpu>();
#endif

    return 0;

}

