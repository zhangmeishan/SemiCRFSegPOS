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
	vector<vector<LookupNode> > word_inputs;
	vector<ConcatNode> token_repsents;

	WindowBuilder word_window;
	vector<UniNode> word_hidden1;

	LSTMBuilder left_lstm;
	LSTMBuilder right_lstm;

	vector<BiNode> word_hidden2;
	vector<SegBuilder> outputseg;
	vector<LinearNode> output;

	NRMat<PNode> poutput; //use to store pointer matrix of outputs
	int max_seg_length;
	int type_num;

	// node pointers
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int maxsegLen, int typeNum){
		max_seg_length = maxsegLen;
		type_num = typeNum;
		int segNum = sent_length * max_seg_length;
		word_inputs.resize(sent_length);
		for (int idx = 0; idx < sent_length; idx++){
			word_inputs[idx].resize(type_num + 1);
		}
		token_repsents.resize(sent_length);
		word_window.resize(sent_length);
		word_hidden1.resize(sent_length);
		left_lstm.resize(sent_length);
		right_lstm.resize(sent_length);
		word_hidden2.resize(sent_length);
		outputseg.resize(segNum);
		for (int idx = 0; idx < segNum; idx++){
			outputseg[idx].resize(maxsegLen);
		}
		output.resize(segNum);
	}

	inline void clear(){
		Graph::clear();
		clearVec(word_inputs);
		token_repsents.clear();
		word_window.clear();
		word_hidden1.clear();
		left_lstm.clear();
		right_lstm.clear();
		word_hidden2.clear();
		outputseg.clear();
		output.clear();		
	}

public:
	inline void initial(LookupTable& words, vector<LookupTable>& types, UniParams& tanh1_project, LSTMParams& left_lstm_project,
		LSTMParams& right_lstm_project, BiParams& tanh2_project, SegParams& seglayer_project, UniParams& olayer_linear, int wordcontext){
		for (int idx = 0; idx < word_inputs.size(); idx++) {
			word_inputs[idx][0].setParam(&words);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].setParam(&types[idy-1]);
			}

			word_hidden1[idx].setParam(&tanh1_project);
			word_hidden1[idx].setFunctions(&tanh, &tanh_deri);

			word_hidden2[idx].setParam(&tanh2_project);
			word_hidden2[idx].setFunctions(&sigmoid, &sigmoid_deri);
		}	
		word_window.setContext(wordcontext);
		left_lstm.setParam(&left_lstm_project, true);
		right_lstm.setParam(&right_lstm_project, false);

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
			word_inputs[idx][0].forward(feature.words[0]);
			execs.push_back(&word_inputs[idx][0]);

			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].forward(feature.types[idy-1]);
				execs.push_back(&word_inputs[idx][idy]);
			}

			token_repsents[idx].forward(getPNodes(word_inputs[idx], word_inputs[idx].size()));
			execs.push_back(&token_repsents[idx]);
		}

		//windowlized
		word_window.forward(getPNodes(token_repsents, seq_size));
		word_window.traverseNodes(execs);

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(&(word_window._outputs[idx]));
			execs.push_back(&word_hidden1[idx]);
		}

		left_lstm.forward(getPNodes(word_hidden1, seq_size));
		left_lstm.traverseNodes(execs);
		right_lstm.forward(getPNodes(word_hidden1, seq_size));
		right_lstm.traverseNodes(execs);

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden2[idx].forward(&(left_lstm._hiddens[idx]), &(right_lstm._hiddens[idx]));
			execs.push_back(&word_hidden2[idx]);
		}

		static int offset;
		vector<PNode> segnodes;
		for (int idx = 0; idx < seq_size; idx++) {
			offset = idx * max_seg_length;
			segnodes.clear();
			for (int dist = 0; idx + dist < seq_size && dist < max_seg_length; dist++) {
				segnodes.push_back(&word_hidden2[idx + dist]);
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
		_types.clear();
	}

	~TNNClassifier() {
		if (_pcg != NULL)
			delete _pcg;
		_pcg = NULL;
	}

public:
	LookupTable _words;
	vector<LookupTable> _types;

	int _wordcontext, _wordwindow;
	int _wordDim;
	vector<int> _typeDims;
	int _unitsize;


	int _hiddensize1;
	int _rnnhiddensize;
	int _hiddensize2;
	int _seghiddensize;
	int _inputsize;
	LSTMParams _left_lstm_project; //left lstm
	LSTMParams _right_lstm_project; //right lstm
	UniParams _tanh1_project; // hidden
	BiParams _tanh2_project; // hidden
	SegParams _seglayer_project; //segmentation
	UniParams _olayer_linear; // output
	

	//SoftMaxLoss _loss;
	SemiCRFMLLoss _loss;

	int _labelSize;

	Metric _eval;

	dtype _dropOut;

	ModelUpdate _ada;

	ComputionGraph *_pcg;

	CheckGrad _checkgrad;

public:
	//embeddings are initialized before this separately.
	inline void init(int wordcontext, int charcontext, int hiddensize1, int rnnhiddensize, int hiddensize2, int seghiddensize, int labelSize) {
		if (_words.nVSize == 0 && _loss.labelSize > 0 && _loss.maxLen > 0){
			std::cout << "Please initialize embeddings before this" << std::endl;
			return;
		}
		_wordcontext = wordcontext;
		_wordwindow = 2 * _wordcontext + 1;
		_wordDim = _words.nDim;
		_unitsize = _wordDim;
		_typeDims.clear();
		for (int idx = 0; idx < _types.size(); idx++){
			_typeDims.push_back(_types[idx].nDim);
			_unitsize += _typeDims[idx];
		}


		_labelSize = labelSize;
		_hiddensize1 = hiddensize1;
		_rnnhiddensize = rnnhiddensize;
		_hiddensize2 = hiddensize2;
		_seghiddensize = seghiddensize;
		_inputsize = _wordwindow * _unitsize;


		_tanh1_project.initial(_hiddensize1, _inputsize, true, 100);
		_left_lstm_project.initial(rnnhiddensize, _hiddensize1, 200);
		_right_lstm_project.initial(rnnhiddensize, _hiddensize1, 300);
		_tanh2_project.initial(_hiddensize1, rnnhiddensize, rnnhiddensize, true, 400);
		_seglayer_project.initial(_seghiddensize, _hiddensize2, 500);
		_olayer_linear.initial(_labelSize, _seghiddensize, false, 600);

		assert(_loss.labelSize == _labelSize);

		//ada
		_words.exportAdaParams(_ada);
		for (int idx = 0; idx < _types.size(); idx++){
			_types[idx].exportAdaParams(_ada);
		}
		_tanh1_project.exportAdaParams(_ada);
		_left_lstm_project.exportAdaParams(_ada);
		_right_lstm_project.exportAdaParams(_ada);
		_tanh2_project.exportAdaParams(_ada);
		_seglayer_project.exportAdaParams(_ada);
		_olayer_linear.exportAdaParams(_ada);
		_loss.exportAdaParams(_ada);


		_pcg = new ComputionGraph();
		_pcg->createNodes(ComputionGraph::max_sentence_length, _loss.maxLen, _types.size());
		_pcg->initial(_words, _types, _tanh1_project, _left_lstm_project, _right_lstm_project, _tanh2_project, _seglayer_project, _olayer_linear, _wordcontext);

		//check grad
		_checkgrad.add(&(_words.E), "_words.E");
		for (int idx = 0; idx < _types.size(); idx++){
			stringstream ss;
			ss << "_types[" << idx << "].E";
			_checkgrad.add(&(_types[idx].E), ss.str());
		}
		_checkgrad.add(&(_tanh1_project.W), "_tanh1_project.W");
		_checkgrad.add(&(_tanh1_project.b), "_tanh1_project.b");

		_checkgrad.add(&(_left_lstm_project.input.W1), "_left_lstm_project.input.W1");
		_checkgrad.add(&(_left_lstm_project.input.W2), "_left_lstm_project.input.W2");
		_checkgrad.add(&(_left_lstm_project.input.W3), "_left_lstm_project.input.W3");
		_checkgrad.add(&(_left_lstm_project.input.b), "_left_lstm_project.input.b");
		_checkgrad.add(&(_left_lstm_project.forget.W1), "_left_lstm_project.forget.W1");
		_checkgrad.add(&(_left_lstm_project.forget.W2), "_left_lstm_project.forget.W2");
		_checkgrad.add(&(_left_lstm_project.forget.W3), "_left_lstm_project.forget.W3");
		_checkgrad.add(&(_left_lstm_project.forget.b), "_left_lstm_project.forget.b");
		_checkgrad.add(&(_left_lstm_project.output.W1), "_left_lstm_project.output.W1");
		_checkgrad.add(&(_left_lstm_project.output.W2), "_left_lstm_project.output.W2");
		_checkgrad.add(&(_left_lstm_project.output.W3), "_left_lstm_project.output.W3");
		_checkgrad.add(&(_left_lstm_project.output.b), "_left_lstm_project.output.b");
		_checkgrad.add(&(_left_lstm_project.cell.W1), "_left_lstm_project.cell.W1");
		_checkgrad.add(&(_left_lstm_project.cell.W2), "_left_lstm_project.cell.W2");
		_checkgrad.add(&(_left_lstm_project.cell.b), "_left_lstm_project.cell.b");

		_checkgrad.add(&(_right_lstm_project.input.W1), "_right_lstm_project.input.W1");
		_checkgrad.add(&(_right_lstm_project.input.W2), "_right_lstm_project.input.W2");
		_checkgrad.add(&(_right_lstm_project.input.W3), "_right_lstm_project.input.W3");
		_checkgrad.add(&(_right_lstm_project.input.b), "_right_lstm_project.input.b");
		_checkgrad.add(&(_right_lstm_project.forget.W1), "_right_lstm_project.forget.W1");
		_checkgrad.add(&(_right_lstm_project.forget.W2), "_right_lstm_project.forget.W2");
		_checkgrad.add(&(_right_lstm_project.forget.W3), "_right_lstm_project.forget.W3");
		_checkgrad.add(&(_right_lstm_project.forget.b), "_right_lstm_project.forget.b");
		_checkgrad.add(&(_right_lstm_project.output.W1), "_right_lstm_project.output.W1");
		_checkgrad.add(&(_right_lstm_project.output.W2), "_right_lstm_project.output.W2");
		_checkgrad.add(&(_right_lstm_project.output.W3), "_right_lstm_project.output.W3");
		_checkgrad.add(&(_right_lstm_project.output.b), "_right_lstm_project.output.b");
		_checkgrad.add(&(_right_lstm_project.cell.W1), "_right_lstm_project.cell.W1");
		_checkgrad.add(&(_right_lstm_project.cell.W2), "_right_lstm_project.cell.W2");
		_checkgrad.add(&(_right_lstm_project.cell.b), "_right_lstm_project.cell.b");

		_checkgrad.add(&(_tanh2_project.W1), "_tanh2_project.W1");
		_checkgrad.add(&(_tanh2_project.W2), "_tanh2_project.W2");
		_checkgrad.add(&(_tanh2_project.b), "_tanh2_project.b");

		_checkgrad.add(&(_seglayer_project.B.W), "_seglayer_project.B.W");
		_checkgrad.add(&(_seglayer_project.B.b), "_seglayer_project.B.b");
		_checkgrad.add(&(_seglayer_project.M.W), "_seglayer_project.M.W");
		_checkgrad.add(&(_seglayer_project.B.b), "_seglayer_project.M.b");
		_checkgrad.add(&(_seglayer_project.E.W), "_seglayer_project.E.W");
		_checkgrad.add(&(_seglayer_project.B.b), "_seglayer_project.E.b");
		_checkgrad.add(&(_seglayer_project.S.W), "_seglayer_project.S.W");
		_checkgrad.add(&(_seglayer_project.B.b), "_seglayer_project.S.b");

		_checkgrad.add(&(_olayer_linear.W), "_olayer_linear.W");
		_checkgrad.add(&(_loss.T), "_loss.T");

		if (_ada._params.size() != _checkgrad._params.size()){
			std::cout << "_ada._params: " << _ada._params.size() << ",  _checkgrad._params: " << _checkgrad._params.size() << std::endl;
		}
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
