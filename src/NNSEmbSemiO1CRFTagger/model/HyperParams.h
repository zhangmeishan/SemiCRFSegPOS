#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{
public:
	//must assign
	int wordContext;
	dtype dropProb;
	int hiddenSize1;
	int rnnHiddenSize;
	int hiddenSize2;
	int segHiddenSize;

	//auto generated
	int wordWindow;
	int wordDim;
	vector<int> typeDims;
	int unitSize;
	vector<int> maxLabelLength;
	int segDim;
	int inputSize;
	int labelSize;
	int maxsegLen;
	int segLabelSize;
	
	// for optimization
	dtype nnRegular, adaAlpha, adaEps;
public:
	HyperParams(){
		bAssigned = false;
	}

	void setReqared(Options& opt){
		wordContext = opt.wordcontext;
		hiddenSize1 = opt.hiddenSize;
		hiddenSize2 = opt.hiddenSize;
		rnnHiddenSize = opt.rnnHiddenSize;
		dropProb = opt.dropProb;
		segHiddenSize = opt.segHiddenSize;

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