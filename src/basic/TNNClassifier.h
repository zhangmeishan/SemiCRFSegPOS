/*
 * TNNClassifier.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_TNNClassifier_H_
#define SRC_TNNClassifier_H_

#include <iostream>

#include "Example.h"
#include "Metric.h"
#include "N3L.h"
#include "Segmentation.h"

using namespace nr;
using namespace std;

struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	vector<LookupNode> word_inputs;
	WindowBuilder word_window;
	vector<UniNode> word_hidden1;
	vector<SegBuilder> outputseg;
	vector<LinearNode> output;

	NRMat<PNode> poutput; //use to store pointer matrix of outputs
	int max_seg_length;

	// node pointers
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int maxsegLen){
		max_seg_length = maxsegLen;
		int segNum = sent_length * max_seg_length;
		word_inputs.resize(sent_length);
		word_window.resize(sent_length);
		word_hidden1.resize(sent_length);
		outputseg.resize(segNum);
		for (int idx = 0; idx < segNum; idx++){
			outputseg[idx].resize(sent_length);
		}
		output.resize(segNum);
	}

	inline void clear(){
		Graph::clear();
		word_inputs.clear();
		word_window.clear();
		word_hidden1.clear();
		outputseg.clear();
		output.clear();
	}

public:
	inline void initial(LookupTable& words, UniParams& tanh_project, SegParams& seglayer_project, UniParams& olayer_linear, int wordcontext){
		for (int idx = 0; idx < word_inputs.size(); idx++) {
			word_inputs[idx].setParam(&words);
			word_hidden1[idx].setParam(&tanh_project);
			word_hidden1[idx].setFunctions(&tanh, &tanh_deri);
		}	
		word_window.setContext(wordcontext);

		for (int idx = 0; idx < output.size(); idx++){
			outputseg[idx].setParam(&seglayer_project);
			outputseg[idx].setFunctions(&tanh, &tanh_deri);
			output[idx].setParam(&olayer_linear);
		}		
	}



public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		clearValue(); // compute is a must step for train, predict and cost computation
		int seq_size = features.size();

		//forward
		// word-level neural networks
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			//input
			word_inputs[idx].forward(feature.words[0]);
			execs.push_back(&word_inputs[idx]);
		}

		//windowlized
		word_window.forward(getPNodes(word_inputs, seq_size));
		word_window.traverseNodes(execs);

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(&(word_window._outputs[idx]));
			execs.push_back(&word_hidden1[idx]);
		}

		static int offset;
		vector<PNode> segnodes;
		for (int idx = 0; idx < seq_size; idx++) {
			offset = idx * max_seg_length;
			segnodes.clear();
			for (int dist = 0; idx + dist < seq_size && dist < max_seg_length; dist++) {
				segnodes.push_back(&word_hidden1[idx + dist]);
				outputseg[offset + dist].forward(segnodes);
				outputseg[offset + dist].traverseNodes(execs);
			}
		}
		
		poutput.resize(seq_size, max_seg_length);
		poutput = NULL;
		offset = 0;
		for (int idx = 0; idx < seq_size; idx++) {
			offset = idx * max_seg_length;
			for (int dist = 0; idx + dist < seq_size && dist < max_seg_length; dist++) {
				output[offset + dist].forward(&(outputseg[offset + dist]._output));
				execs.push_back(&output[offset + dist]);
				poutput[idx][dist] = &output[offset + dist];
			}
		}
	}

};

//A native neural network classfier using only word embeddings

class TNNClassifier {
public:
	TNNClassifier() {
		_dropOut = 0.0;
		_pcg = NULL;
	}

	~TNNClassifier() {
		if (_pcg != NULL)
			delete _pcg;
		_pcg = NULL;
	}

public:
	LookupTable _words;

	int _wordcontext, _wordwindow;
	int _wordSize;
	int _wordDim;


	int _hiddensize;
	int _seghiddensize;
	int _inputsize;

	UniParams _tanh_project; // hidden
	SegParams _seglayer_project; //segmentation
	UniParams _olayer_linear; // output
	

	//SoftMaxLoss _loss;
	Semi0CRFMLLoss _loss;

	int _labelSize;

	Metric _eval;

	dtype _dropOut;

	ModelUpdate _ada;

	ComputionGraph *_pcg;

	CheckGrad _checkgrad;

public:
	//embeddings are initialized before this separately.
	inline void init(int wordcontext, int charcontext, int hiddensize, int seghiddensize, int labelSize) {
		if (_words.nVSize == 0 && _loss.labelSize > 0 && _loss.maxLen > 0){
			std::cout << "Please initialize embeddings before this" << std::endl;
			return;
		}
		_wordcontext = wordcontext;
		_wordwindow = 2 * _wordcontext + 1;
		_wordDim = _words.nDim;


		_labelSize = labelSize;
		_hiddensize = hiddensize;
		_seghiddensize = seghiddensize;
		_inputsize = _wordwindow * _wordDim;


		_tanh_project.initial(_hiddensize, _inputsize, true, 100);
		_seglayer_project.initial(_seghiddensize, _hiddensize, 200);
		_olayer_linear.initial(_labelSize, _seghiddensize, false, 300);

		assert(_loss.labelSize == _labelSize);

		//ada
		_words.exportAdaParams(_ada);
		_tanh_project.exportAdaParams(_ada);
		_seglayer_project.exportAdaParams(_ada);
		_olayer_linear.exportAdaParams(_ada);


		_pcg = new ComputionGraph();
		_pcg->createNodes(ComputionGraph::max_sentence_length, _loss.maxLen);
		_pcg->initial(_words, _tanh_project, _seglayer_project, _olayer_linear, _wordcontext);

		//check grad
		_checkgrad.add(&(_words.E), "_words.E");
		_checkgrad.add(&(_tanh_project.W), "_tanh_project.W");
		_checkgrad.add(&(_tanh_project.b), "_tanh_project.b");
		_checkgrad.add(&(_seglayer_project.B.W), "_seglayer_project.B.W");
		_checkgrad.add(&(_seglayer_project.B.b), "_seglayer_project.B.b");
		_checkgrad.add(&(_seglayer_project.M.W), "_seglayer_project.M.W");
		_checkgrad.add(&(_seglayer_project.B.b), "_seglayer_project.M.b");
		_checkgrad.add(&(_seglayer_project.E.W), "_seglayer_project.E.W");
		_checkgrad.add(&(_seglayer_project.B.b), "_seglayer_project.E.b");
		_checkgrad.add(&(_seglayer_project.S.W), "_seglayer_project.S.W");
		_checkgrad.add(&(_seglayer_project.B.b), "_seglayer_project.S.b");
		_checkgrad.add(&(_olayer_linear.W), "_olayer_linear.W");
	}


	inline dtype train(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;

		static vector<PMat> tpmats;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			//forward
			_pcg->forward(example.m_features, true); 

			//loss function
			int seq_size = example.m_features.size();
			//for (int idx = 0; idx < seq_size; idx++) {
				//cost += _loss.loss(&(_pcg->output[idx]), example.m_labels[idx], _eval, example_num);				
			//}
			cost += _loss.loss(_pcg->poutput, example.m_seglabels, _eval, example_num);

			// backward, which exists only for training 
			_pcg->backward();
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const vector<Feature>& features, NRMat<int>& results) {
		_pcg->forward(features);
		int seq_size = features.size();
		//results.resize(seq_size);
		//for (int idx = 0; idx < seq_size; idx++) {
		//	_loss.predict( &(_pcg->output[idx]), results[idx]);
		//}
		_loss.predict(_pcg->poutput, results);
	}

	inline dtype cost(const Example& example){
		_pcg->forward(example.m_features); //forward here

		int seq_size = example.m_features.size();

		dtype cost = 0.0;
		//loss function
		//for (int idx = 0; idx < seq_size; idx++) {
		//	cost += _loss.cost(&(_pcg->output[idx]), example.m_labels[idx], 1);
		//}
		cost += _loss.cost(_pcg->poutput, example.m_seglabels, 1);

		return cost;
	}


	void updateModel() {
		_ada.update();
	}

	void checkgrad(const vector<Example>& examples, int iter){
		ostringstream out;
		out << "Iteration: " << iter;
		_checkgrad.check(this, examples, out.str());
	}

	void writeModel();

	void loadModel();



public:
	inline void resetEval() {
		_eval.reset();
	}

	inline void setDropValue(dtype dropOut) {
		_dropOut = dropOut;
	}

	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /* SRC_TNNClassifier_H_ */
