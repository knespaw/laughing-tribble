#include <iostream>

#include "llama.h"

#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config.h"

#include "spdlog/fmt/fmt.h"

#include "utils/logger.h"



constexpr std::string_view PROMPT_INSERT = "<PROMPT>";



enum TokenSampler
{
	GREEDY = 0,
};

constexpr std::string_view to_string(const TokenSampler sampler) noexcept
{
	switch (sampler)
	{
	case GREEDY:
		return "GREEDY";
	}

	throw std::invalid_argument("unrecognized token sampler");
}


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

std::string InferenceParameters::print() const
{
	fmt::memory_buffer buffer;

	fmt::format_to(
		std::back_inserter(buffer),
		"InferenceParameters{{modelFile=\"{}\", maxParallelSequences={}, "
		"maxTotalBatchSize={}, totalContextSize={}, nCPUThreads={}, "
		"nGPUStoredLayers={}, tokenSampler={}}}",
		modelFile != nullptr ? modelFile : "(null)",
		maxParallelSequences,
		maxTotalBatchSize,
		totalContextSize,
		nCPUThreads,
		nGPUStoredLayers,
		to_string(tokenSampler)
	);

	return fmt::to_string(buffer);
}


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

Tokenizer::Tokenizer(std::string promptBlueprint, const llama_model *model) :
	promptTemplate(std::move(promptBlueprint)),
	_promptInsertIdx(promptTemplate.find(PROMPT_INSERT))
{
	if (_promptInsertIdx == std::string::npos)
	{
		LOG_ERROR(
			"`Tokenizer` initialization failed. invalid prompt template "
			"\"{}\". should contain \"{}\" inside",
			promptTemplate,
			PROMPT_INSERT
		);
		throw std::invalid_argument("invalid prompt template");
	}

	if (const llama_vocab *vocab = llama_model_get_vocab(model);
		vocab == nullptr)
	{
		LOG_ERROR(
			"`Tokenizer` initialization failed. cannot load model vocabulary"
		);
		throw std::runtime_error("failed to load vocabulary");
	}
	else
	{
		// initial capacity
		tokens	   = std::vector(3 * promptTemplate.size(), 0);
		vocabulary = vocab;
	}
}

std::string Tokenizer::preparePrompt(const std::string &input) const
{
	std::string prompt;

	prompt.reserve(promptTemplate.size() - PROMPT_INSERT.size() + input.size());
	prompt.append(promptTemplate, 0, _promptInsertIdx);
	prompt.append(input);
	prompt.append(
		promptTemplate,
		_promptInsertIdx + PROMPT_INSERT.size(),
		std::string::npos
	);

	return prompt;
}

void Tokenizer::tokenize(const std::string &text)
{
	LOG_DEBUG("starting tokenization of \"{}\"", text);

	const auto prompt = preparePrompt(text);

	LOG_DEBUG("prepared model prompt \"{}\"", prompt);

	// +2 for some safety margin
	tokens.resize(prompt.size() + 2, 0);

	const auto n_tokens = llama_tokenize(
		vocabulary,
		prompt.c_str(),
		prompt.size(),
		tokens.data(),
		tokens.size(),
		true,
		true
	);

	if (n_tokens < 0)
	{
		LOG_ERROR("tokenization failed for prompt \"{}\"", prompt);
		throw std::runtime_error("failed to tokenize prompt");
	}

	LOG_DEBUG("tokenization finished. {} tokens were prepared", n_tokens);

	tokens.resize(n_tokens);
}

llama_token Tokenizer::getToken(const int position) const
{
	return tokens[position];
}

size_t Tokenizer::nTokens() const noexcept { return tokens.size(); }

const llama_vocab *Tokenizer::getVocab() const noexcept { return vocabulary; }


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

Generator::Generator(const InferenceParameters &params)
{
	LOG_INFO("creating new `Generator` instance with\n{}", params.print());

	LOG_DEBUG("loading model from \"{}\"", params.modelFile);

	llama_model_params model_params = llama_model_default_params();

	model_params.n_gpu_layers = params.nGPUStoredLayers;

	llama_model *m = llama_model_load_from_file(params.modelFile, model_params);

	if (m == nullptr)
	{
		LOG_ERROR("`Generator` initialization failed. cannot load model");
		throw std::runtime_error("failed to load model from file");
	}

	LOG_DEBUG("starting new inference context");

	llama_context_params ctx_params = llama_context_default_params();

	ctx_params.n_ctx	 = params.totalContextSize;
	ctx_params.n_threads = params.nCPUThreads;
	ctx_params.n_batch	 = params.maxTotalBatchSize;
	ctx_params.n_seq_max = params.maxParallelSequences;

	llama_context *c = llama_init_from_model(m, ctx_params);

	if (c == nullptr)
	{
		LOG_ERROR(
			"`Generator` initialization failed. cannot initialize context"
		);
		llama_model_free(m);
		throw std::runtime_error("failed to initialize context");
	}

	LOG_DEBUG("starting new token sampler");

	llama_sampler *s = nullptr;
	switch (params.tokenSampler)
	{
	case GREEDY:
		s = llama_sampler_init_greedy();
		break;
	}

	if (s == nullptr)
	{
		LOG_ERROR(
			"`Generator` initialization failed. cannot initialize sampler"
		);
		llama_free(c);
		llama_model_free(m);
		throw std::runtime_error("failed to initialize sampler");
	}

	model	= m;
	ctx		= c;
	sampler = s;

	LOG_INFO("`Generator` initialized successfully");
}

Generator::~Generator()
{
	llama_sampler_free(sampler);
	llama_free(ctx);
	llama_model_free(model);
}

const llama_model *Generator::getModel() const noexcept { return model; }

llama_context *Generator::getContext() const noexcept { return ctx; }

llama_sampler *Generator::getSampler() const noexcept { return sampler; }


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

Inference::Inference(const InferenceParameters &params) :
	generator(params),
	tokenizer(params.promptTemplate, generator.getModel()),
	params(params)
{
}

llama_batch Inference::prepareBatch(const std::string &input)
{
	tokenizer.tokenize(input);

	LOG_DEBUG("preparing new batch");

	// TODO: params
	llama_batch batch =
		llama_batch_init(params.batchSize, 0, params.maxParallelSequences);

	for (int i = 0; i < tokenizer.nTokens(); i++)
	{
		batch.token[i]	   = tokenizer.getToken(i);
		batch.pos[i]	   = i;
		batch.n_seq_id[i]  = 1; // todo
		batch.seq_id[i][0] = 0; // todo
		// only output of the last token is taken into account
		batch.logits[i] = false;
	}

	// to get probability of the last token
	batch.logits[tokenizer.nTokens() - 1] = true;
	batch.n_tokens						  = tokenizer.nTokens();

	return batch;
}

std::string Inference::infer(const std::string &input)
{
	LOG_DEBUG("running new inference for \'{}\'", input);

	llama_batch batch = prepareBatch(input);

	// PREFILL
	LOG_DEBUG("running prefill stage");

	// TODO
	switch (llama_decode(generator.getContext(), batch))
	{
	case 0:
		break;
	default:
		LOG_ERROR("inference failed. prefill stage errored");
		llama_batch_free(batch);
		return "";
	}

	// DECODE
	LOG_DEBUG("running decode stage");

	std::string response;
	response.reserve(1000); // todo

	int currPos = batch.n_tokens;

	while (true)
	{
		const llama_token newTokenId = llama_sampler_sample(
			generator.getSampler(),
			generator.getContext(),
			-1
		);

		llama_sampler_accept(generator.getSampler(), newTokenId);

		if (llama_vocab_is_eog(tokenizer.getVocab(), newTokenId))
		{
			break;
		}

		// usually, tokens represent 3-4 characters, but some may be longer,
		// thus 128 is a safe bet
		char buffer[128];

		const auto nBytes = llama_token_to_piece(
			tokenizer.getVocab(),
			newTokenId,
			buffer,
			sizeof(buffer),
			0,
			false
		);

		if (nBytes > 0)
		{
			response.append(buffer, nBytes);
			std::cout.write(buffer, nBytes);
			std::cout.flush();
		}

		// ??
		batch.n_tokens	   = 0;
		batch.token[0]	   = newTokenId;
		batch.pos[0]	   = currPos;
			batch.n_seq_id[0]  = 1;
		batch.seq_id[0][0] = 0;
		batch.logits[0]	   = true;
		batch.n_tokens	   = 1;

		// TODO
		switch (llama_decode(generator.getContext(), batch))
		{
		case 0:
			break;
		default:
			LOG_ERROR("inference failed. failure during decode phase");
			llama_batch_free(batch);
			return response;
		}

		currPos++;
	}

	llama_batch_free(batch);

	return response;
}
