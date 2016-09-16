#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{
	// must assign
	int wordContext;
	dtype dropProb;
	int hiddenSize;
	int rnnHiddenSize;
	
	//auto generated
	int wordWindow;
	int wordDim;
	vector<int> typeDims;
	int unitSize;
	int inputSize;
	int labelSize;
	vector<int> maxLabelLength;
	int maxsegLen;

	// for optimization
	dtype nnRegular, adaAlpha, adaEps;

public:
	HyperParams(){
		bAssigned = false;
	}

	void setReqared(Options& opt){
		wordContext = opt.wordcontext;
		hiddenSize = opt.hiddenSize;
		rnnHiddenSize = opt.rnnHiddenSize;
		dropProb = opt.dropProb;
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bVaild(){
		return bAssigned;
	}

	void print(){}

private:
	bool bAssigned;
};
#endif /*SRC_HyperParams_H_*/