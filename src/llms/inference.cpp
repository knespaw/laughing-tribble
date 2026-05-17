#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "llama.h"
#include "spdlog/fmt/fmt.h"
#include <boost/uuid.hpp>

#include "utils/logger.h"



constexpr std::string_view USR_PROMPT = "<USR_PROMPT>";



class GenerationError : public std::runtime_error
{
	private:

		int error_code_;

	public:

		GenerationError(const int code, const std::string &msg) :
			std::runtime_error(msg),
			error_code_(code)
		{
		}

		[[nodiscard]] int getErrorCode() const { return error_code_; }
};



struct SequenceMeta
{
		llama_seq_id sequenceId;
		llama_pos	 position;

		[[nodiscard]] bool		  initialized() const;
		void					  restart();
		[[nodiscard]] std::string info() const;
};


bool SequenceMeta::initialized() const { return position > 0; }

void SequenceMeta::restart() { position = 0; }

std::string SequenceMeta::info() const
{
	fmt::memory_buffer buffer;

	fmt::format_to(
		std::back_inserter(buffer),
		"sequenceID : {}, tokenPosition : {}",
		sequenceId,
		position
	);

	return fmt::to_string(buffer);
}



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
		uint32_t		  totalContextSize;
		int32_t			  nCPUThreads;
		int32_t			  nGPUStoredLayers;
		DecodingStrategy  decodingStrategy;
		const std::string promptTemplate;

		[[nodiscard]] std::string print() const;
};

std::string InferenceParameters::print() const
{
	fmt::memory_buffer buffer;

	fmt::format_to(
		std::back_inserter(buffer),
		"InferenceParameters{{modelFile=\"{}\", maxParallelSequences={}, "
		"maxTotalBatchSize={}, totalContextSize={}, "
		"nCPUThreads={}, "
		"nGPUStoredLayers={}, decodingStrategy={}, promptTemplate={}}}",
		modelFile != nullptr ? modelFile : "(null)",
		maxParallelSequences,
		maxTotalBatchSize,
		totalContextSize,
		nCPUThreads,
		nGPUStoredLayers,
		static_cast<int>(decodingStrategy),
		promptTemplate
	);

	return fmt::to_string(buffer);
}



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


llama_model *Inference::loadModel(const InferenceParameters &params)
{
	LOG_DEBUG("loading model from \"{}\"", params.modelFile);

	llama_model_params model_params = llama_model_default_params();

	model_params.n_gpu_layers = params.nGPUStoredLayers;

	llama_model *model =
		llama_model_load_from_file(params.modelFile, model_params);

	if (model == nullptr)
	{
		LOG_ERROR("cannot load model from \"{}\"", params.modelFile);
		throw std::runtime_error("failed to load model from file");
	}

	LOG_DEBUG("model loaded from \"{}\"", params.modelFile);

	return model;
}

llama_context *
Inference::startContext(const InferenceParameters &params, llama_model *model)
{
	LOG_DEBUG("starting new inference context");

	llama_context_params ctx_params = llama_context_default_params();

	ctx_params.n_ctx	 = params.totalContextSize;
	ctx_params.n_threads = params.nCPUThreads;
	ctx_params.n_batch	 = params.maxTotalBatchSize;
	ctx_params.n_seq_max = params.maxParallelSequences;

	llama_context *ctx = llama_init_from_model(model, ctx_params);

	if (ctx == nullptr)
	{
		LOG_ERROR("cannot start new context");
		llama_model_free(model);
		throw std::runtime_error("failed to initialize context");
	}

	LOG_DEBUG("inference context started");

	return ctx;
}

llama_sampler *Inference::initSampler(
	const InferenceParameters &params,
	llama_context			  *ctx,
	llama_model				  *model
)
{
	LOG_DEBUG(
		"initializing sampler for \"{}\" decoding strategy",
		static_cast<int>(params.decodingStrategy)
	);

	llama_sampler *sampler = nullptr;

	switch (params.decodingStrategy)
	{
	case GREEDY:
		sampler = llama_sampler_init_greedy();
		break;
	}

	if (sampler == nullptr)
	{
		LOG_DEBUG(
			"failed to initialize sampler for \"{}\" decoding strategy",
			static_cast<int>(params.decodingStrategy)
		);
		llama_free(ctx);
		llama_model_free(model);
		throw std::runtime_error("failed to initialize sampler");
	}

	LOG_DEBUG(
		"sampler initialized for \"{}\" decoding strategy",
		static_cast<int>(params.decodingStrategy)
	);

	return sampler;
}

const llama_vocab *Inference::loadVocabulary(llama_model *model)
{
	LOG_DEBUG("loading model vocabulary");

	if (const llama_vocab *vocab = llama_model_get_vocab(model);
		vocab == nullptr)
	{
		LOG_ERROR("cannot load model vocabulary");
		llama_model_free(model);
		throw std::runtime_error("failed to load vocabulary");
	}
	else
	{
		LOG_DEBUG("loaded model vocabulary");
		return vocab;
	}
}


Inference::Inference(const InferenceParameters &params) :
	model_(loadModel(params)),
	vocab_(loadVocabulary(model_)),
	ctx_(startContext(params, model_)),
	sampler_(initSampler(params, ctx_, model_)),
	parameters_(params),
	tokens_(std::vector(3 * params.promptTemplate.size(), 0)),
	promptInsertIdx_(params.promptTemplate.find(USR_PROMPT))
{
	if (promptInsertIdx_ == std::string::npos)
	{
		LOG_ERROR(
			"`Inference` initialization failed. invalid prompt template "
			"\"{}\". should contain \"{}\" inside",
			params.promptTemplate,
			USR_PROMPT
		);
		throw std::invalid_argument("invalid prompt template");
	}

	LOG_INFO(
		"successfully created new `Inference` instance with:\n{}",
		params.print()
	);
}

Inference::~Inference()
{
	LOG_INFO("destructing `Inference` instance");

	llama_sampler_free(sampler_);
	llama_free(ctx_);
	llama_model_free(model_);
}


std::string Inference::preparePrompt(const Input &inp) const
{
	std::string prompt;
	prompt.reserve(
		parameters_.promptTemplate.size() - USR_PROMPT.size() +
		inp.prompt.size()
	);

	prompt.append(parameters_.promptTemplate, 0, promptInsertIdx_);
	prompt.append(inp.prompt);
	prompt.append(
		parameters_.promptTemplate,
		promptInsertIdx_ + USR_PROMPT.size(),
		std::string::npos
	);

	return prompt;
}

void Inference::tokenize(const std::string &text)
{
	LOG_DEBUG("starting tokenization of \"{}\"", text);

	// +2 for some safety margin
	tokens_.resize(text.size() + 2, 0);

	const auto nTokens = llama_tokenize(
		vocab_,
		text.c_str(),
		text.size(),
		tokens_.data(),
		tokens_.size(),
		true,
		true
	);

	if (nTokens < 0)
	{
		LOG_ERROR("tokenization failed for \"{}\"", text);
		throw std::runtime_error("failed to tokenize text");
	}

	LOG_DEBUG("tokenization finished. {} tokens were prepared", nTokens);

	tokens_.resize(nTokens);
}

void Inference::generate(const llama_batch &batch) const
{
	auto const code = llama_decode(ctx_, batch);

	if (code == 0)
	{
		return;
	}

	LOG_ERROR("generation failed with {} code", code);

	// TODO
	switch (code)
	{
	case -1:
		throw GenerationError(code, "invalid input batch");
	case 1:
		throw GenerationError(code, "not enough memory");
	case 2:
		throw GenerationError(code, "aborted");
	default:
		throw GenerationError(code, "fatal");
	}
}

void Inference::prefill(llama_batch &batch, Input &inp) const
{
	LOG_DEBUG(
		"{} :: {} :: running prefill stage for {} tokens",
		inp.id,
		inp.meta.info(),
		batch.n_tokens
	);

	for (int i = 0; i < tokens_.size(); i++)
	{
		// only output of the last token is taken into account
		batch.logits[i]	   = false;
		batch.token[i]	   = tokens_[i];
		batch.pos[i]	   = inp.meta.position + i;
		batch.n_seq_id[i]  = 1;					  // todo
		batch.seq_id[i][0] = inp.meta.sequenceId; // todo
	}

	// get probability of the last token
	batch.logits[tokens_.size() - 1] = true;
	batch.n_tokens					 = tokens_.size();

	try
	{
		generate(batch);
	}
	catch (const GenerationError &err)
	{
		// TODO
	}

	inp.meta.position += batch.n_tokens;

	LOG_DEBUG(
		"{} :: {} :: prefill stage completed",
		inp.id,
		inp.meta.info(),
		batch.n_tokens
	);

	llama_batch_free(batch);
}

llama_token
Inference::sample(std::string *output, const bool consolePrint) const
{
	// TODO
	// last (new) token
	const llama_token newTokenId = llama_sampler_sample(sampler_, ctx_, -1);

	llama_sampler_accept(sampler_, newTokenId);

	if (llama_vocab_is_eog(vocab_, newTokenId))
	{
		return newTokenId;
	}

	// usually, tokens represent 3-4 characters, but some may be longer,
	// thus 128 is a safe bet
	char buffer[128];

	const auto nBytes = llama_token_to_piece(
		vocab_,
		newTokenId,
		buffer,
		sizeof(buffer),
		0,
		false
	);

	if (nBytes > 0)
	{
		output->append(buffer, nBytes);

		if (consolePrint)
		{
			std::cout.write(buffer, nBytes);
			std::cout.flush();
		}
	}

	return newTokenId;
}

Output Inference::decode(llama_batch &batch, Input &inp) const
{
	LOG_DEBUG("{} :: {} :: running decode stage", inp.id, inp.meta.info());

	auto output = Output{inp.id};
	output.response.reserve(500);

	for (auto tokenId = sample(&output.response, true);
		 !llama_vocab_is_eog(vocab_, tokenId);
		 tokenId = sample(&output.response, true))
	{
		// TODO
		batch.n_tokens	   = 1;
		batch.token[0]	   = tokenId;
		batch.pos[0]	   = inp.meta.position;
		batch.n_seq_id[0]  = 1;
		batch.seq_id[0][0] = inp.meta.sequenceId;
		batch.logits[0]	   = true;

		try
		{
			generate(batch);
		}
		catch (const GenerationError &err)
		{
			// TODO
		}

		inp.meta.position++;
	}

	LOG_DEBUG(
		"{} :: {} :: generation finished: \"{}\"",
		output.id,
		inp.meta.info(),
		output.response
	);

	llama_batch_free(batch);

	return output;
}

Output Inference::run(Input &inp)
{
	if (!inp.meta.initialized())
	{
		const auto prompt = preparePrompt(inp);

		tokenize(prompt);

		// TODO
		auto prefillBatch = llama_batch_init(
			tokens_.size(),
			0,
			parameters_.maxParallelSequences
		);

		prefill(prefillBatch, inp);
	}

	auto decodeBatch = llama_batch_init(1, 0, 1);

	return decode(decodeBatch, inp);
}

void Inference::clearMemory() const
{
	llama_memory_clear(llama_get_memory(ctx_), false);
}
