#pragma once


#include <string>
#include <vector>

#include "llama.h"



enum TokenSampler
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
		TokenSampler	  tokenSampler;
		const std::string promptTemplate;

		[[nodiscard]] std::string print() const;
};


class Tokenizer
{
	public:

		Tokenizer(std::string promptBlueprint, const llama_model *model);

		void							 tokenize(const std::string &text);
		[[nodiscard]] llama_token		 getToken(int position) const;
		[[nodiscard]] size_t			 nTokens() const noexcept;
		[[nodiscard]] const llama_vocab *getVocab() const noexcept;

	private:

		const std::string		 promptTemplate;
		const size_t			 _promptInsertIdx;
		const llama_vocab		*vocabulary = nullptr;
		std::vector<llama_token> tokens;

		[[nodiscard]] std::string preparePrompt(const std::string &input) const;
};


class Generator
{
	public:

		explicit Generator(const InferenceParameters &params);
		~Generator();

		[[nodiscard]] const llama_model *getModel() const noexcept;
		[[nodiscard]] llama_context		*getContext() const noexcept;
		[[nodiscard]] llama_sampler		*getSampler() const noexcept;

	private:

		llama_context *ctx	   = nullptr;
		llama_model	  *model   = nullptr;
		llama_sampler *sampler = nullptr;
};


class Inference
{
	public:

		explicit Inference(const InferenceParameters &params);

		[[nodiscard]] std::string infer(const std::string &input);

	private:

		Generator			generator;
		Tokenizer			tokenizer;
		InferenceParameters params;

		llama_batch prepareBatch(const std::string &input);
};
