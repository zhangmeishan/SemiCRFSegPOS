/*
 * SegOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef SEGBuilder_H_
#define SEGBuilder_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Pooling.h"
#include "UniOP.h"

struct SegParams {
	UniParams B;
	UniParams M;
	UniParams E;
	UniParams S;

	SegParams() {
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		B.exportAdaParams(ada);
		M.exportAdaParams(ada);
		E.exportAdaParams(ada);
		S.exportAdaParams(ada);
	}

	inline void initial(int nOSize, int nISize, int seed = 0) {
		B.initial(nOSize, nISize, true, seed);
		M.initial(nOSize, nISize, true, seed + 1);
		E.initial(nOSize, nISize, true, seed + 2);
		S.initial(nOSize, nISize, true, seed + 3);
	}
};

// we can rewrite it as one node, but many duplicated codes
class SegBuilder : NodeBuilder{
public:
	SegParams* _param;

	int _nSize;
	int _inDim;
	int _outDim;

	MaxPoolNode _output;
	vector<UniNode> _tnodes;

public:
	SegBuilder(){

	}

	~SegBuilder(){

	}

	inline void setParam(SegParams* paramInit) {
		_param = paramInit;
		_inDim = _param->B.W.inDim();
		_outDim = _param->B.W.outDim();
	}

	inline void setFunctions(Mat(*f)(const Mat&),
		Mat(*f_deri)(const Mat&, const Mat&)) {
		for (int idx = 0; idx < _tnodes.size(); idx++){
			_tnodes[idx].setFunctions(f, f_deri);
		}
	}

	inline void resize(int maxsize){
		_tnodes.resize(maxsize);
	}

	inline void clear(){
		_tnodes.clear();
	}

public:

	inline void forward(const vector<PNode>& x){
		if (x.size() == 0){
			std::cout << "empty inputs for seg operation" << std::endl;
			return;
		}

		_nSize = x.size();
		if (x[0]->val.rows() != _inDim){
			std::cout << "input dim does not match for seg operation" << std::endl;
			return;
		}

		if (_nSize == 1){
			_tnodes[0].setParam(&(_param->S));
		}
		else{
			_tnodes[0].setParam(&(_param->B));
			for (int idx = 1; idx < _nSize - 1; idx++){
				_tnodes[idx].setParam(&(_param->M));
			}
			_tnodes[_nSize-1].setParam(&(_param->E));
		}

		for (int idx = 0; idx < _nSize; idx++){
			_tnodes[idx].forward(x[idx]);
		}

		_output.forward(getPNodes(_tnodes, _nSize));
	}

	inline void traverseNodes(vector<PNode> &exec){
		for (int idx = 0; idx < _nSize; idx++){
			exec.push_back(&_tnodes[idx]);
		}
		exec.push_back(&_output);
	}

};

#endif /* SEGBuilder_H_ */
