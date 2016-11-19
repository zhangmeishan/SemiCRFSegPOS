/*
 * Tagger.h
 *
 *  Created on: Mar 16, 2015
 *      Author: mszhang
 */

#ifndef SRC_NNSemiCRFTagger_H_
#define SRC_NNSemiCRFTagger_H_


#include "N3L.h"
#include "Driver.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"


#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Tagger {


public:
	unordered_set<string> ignoreLabels;
	unordered_map<string, int> m_feat_stats;
	unordered_map<string, int> m_word_stats;
	unordered_map<string, int> m_char_stats;
	vector<unordered_map<string, int> > m_type_stats;

public:
	Options m_options;

	Pipe m_pipe;

	Driver m_driver;


public:
	Tagger(int memsize);
	virtual ~Tagger();

public:

	int createAlphabet(const vector<Instance>& vecTrainInsts);
	int addTestAlpha(const vector<Instance>& vecInsts);


	void extractLinearFeatures(vector<string>& features, const Instance* pInstance, int idx);
	void extractFeature(Feature& feat, const Instance* pInstance, int idx);

	void convert2Example(const Instance* pInstance, Example& exam);
	void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

public:
	void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
	int predict(const vector<Feature>& features, vector<string>& outputs);
	void test(const string& testFile, const string& outputFile, const string& modelFile);

	void writeModelFile(const string& outputModelFile);
	void loadModelFile(const string& inputModelFile);

};

#endif /* SRC_NNSemiCRFTagger_H_ */
