#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "BMESSegmentation.h"

struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	vector<vector<LookupNode> > word_inputs;
	vector<ConcatNode> token_repsents;

	WindowBuilder word_window;
	vector<UniNode> word_hidden1;

	LSTM1Builder left_lstm;
	LSTM1Builder right_lstm;

	vector<BiNode> word_hidden2;
	vector<LinearNode> output;

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
	inline void createNodes(int sent_length, int typeNum){
		type_num = typeNum;
		resizeVec(word_inputs, sent_length, type_num + 1);
		token_repsents.resize(sent_length);
		word_window.resize(sent_length);
		word_hidden1.resize(sent_length);
		left_lstm.resize(sent_length);
		right_lstm.resize(sent_length);
		word_hidden2.resize(sent_length);
		output.resize(sent_length);
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
		output.clear();
	}

public:
	inline void initial(ModelParams& model_params, HyperParams& hyper_params, AlignedMemoryPool* mem = NULL){
		for (int idx = 0; idx < word_inputs.size(); idx++) {
			word_inputs[idx][0].setParam(&model_params._words);
			word_inputs[idx][0].init(hyper_params.wordDim, hyper_params.dropProb, mem);
			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].setParam(&model_params._types[idy - 1]);
				word_inputs[idx][idy].init(hyper_params.wordDim, hyper_params.dropProb, mem);
			}

			word_hidden1[idx].setParam(&model_params._tanh1_project);
			word_hidden1[idx].init(hyper_params.hiddenSize, hyper_params.dropProb, mem);

			word_hidden2[idx].setParam(&model_params._tanh2_project);
			word_hidden2[idx].init(hyper_params.hiddenSize, hyper_params.dropProb, mem);
			token_repsents[idx].init(hyper_params.unitSize, -1, mem);
		}
		word_window.init(hyper_params.unitSize, hyper_params.wordContext, mem);
		left_lstm.init(&model_params._left_lstm_project, hyper_params.dropProb, true, mem);
		right_lstm.init(&model_params._right_lstm_project, hyper_params.dropProb, false, mem);

		for (int idx = 0; idx < output.size(); idx++){
			output[idx].setParam(&model_params._olayer_linear);
			output[idx].init(hyper_params.labelSize, -1, mem);
		}
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		clearValue(bTrain); // compute is a must step for train, predict and cost computation
		int seq_size = features.size();

		//forward
		// word-level neural networks
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			//input
			word_inputs[idx][0].forward(this, feature.words[0]);

			for (int idy = 1; idy < word_inputs[idx].size(); idy++){
				word_inputs[idx][idy].forward(this, feature.types[idy - 1]);
			}

			token_repsents[idx].forward(this, getPNodes(word_inputs[idx], word_inputs[idx].size()));
		}

		//windowlized
		word_window.forward(this, getPNodes(token_repsents, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden1[idx].forward(this, &(word_window._outputs[idx]));

		}

		left_lstm.forward(this, getPNodes(word_hidden1, seq_size));
		right_lstm.forward(this, getPNodes(word_hidden1, seq_size));

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			word_hidden2[idx].forward(this, &(left_lstm._hiddens[idx]), &(right_lstm._hiddens[idx]));

			output[idx].forward(this, &(word_hidden2[idx]));
		}
	}

};

#endif/*SRC_ComputionGraph_H_*/