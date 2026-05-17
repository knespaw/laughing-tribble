#pragma once


#include <string>
#include <vector>

#include "llama.h"



struct SequenceMeta
{
		llama_seq_id sequenceId;
		llama_pos	 position;

		[[nodiscard]] bool		  initialized() const;
		void					  restart();
		[[nodiscard]] std::string info() const;
};



struct Input
{
		// boost::uuids::uuid id;
		const std::string id;
		std::string		  prompt;
		SequenceMeta	  meta;
};



struct Output
{
		// boost::uuids::uuid id;
		const std::string id;
		std::string		  response;
};



enum DecodingStrategy
{
	GREEDY = 0,
};



struct InferenceParameters
{
		const char		 *modelFile;
		uint32_t		  maxParallelSequences;
		uint32_t		  maxTotalBatchSize;
		int32_t			  batchSize;
		uint32_t		  totalContextSize;
		int32_t			  nCPUThreads;
		int32_t			  nGPUStoredLayers;
		DecodingStrategy  decodingStrategy;
		const std::string promptTemplate;

		[[nodiscard]] std::string print() const;
};



class Inference
{
	public:

		explicit Inference(const InferenceParameters &params);
		~Inference();

		static llama_model		 *loadModel(const InferenceParameters &params);
		static const llama_vocab *loadVocabulary(llama_model *model);
		static llama_context *
		startContext(const InferenceParameters &params, llama_model *model);
		static llama_sampler *initSampler(
			const InferenceParameters &params,
			llama_context			  *ctx,
			llama_model				  *model
		);

		Output run(Input &inp);
		void   clearMemory() const;

	private:

		llama_model		  *model_	= nullptr;
		const llama_vocab *vocab_	= nullptr;
		llama_context	  *ctx_		= nullptr;
		llama_sampler	  *sampler_ = nullptr;

		InferenceParameters		 parameters_;
		std::vector<llama_token> tokens_;
		const size_t			 promptInsertIdx_;

		[[nodiscard]] std::string preparePrompt(const Input &inp) const;
		void					  tokenize(const std::string &text);
		void					  generate(const llama_batch &batch) const;
		void					  prefill(llama_batch &batch, Input &inp) const;
		[[nodiscard]] llama_token
		sample(std::string *output, bool consolePrint) const;
		[[nodiscard]] Output decode(llama_batch &batch, Input &inp) const;
};
