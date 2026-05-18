#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"
#include "spdlog/fmt/fmt.h"
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_hash.hpp>

#include "utils.h"
#include "utils/logger.h"

#include "text.h"

#include "inference.h"



namespace Inference
{
	class GenerationError : public std::runtime_error
	{
			int error_code_;

		public:

			GenerationError(const int code, const std::string &msg) :
				std::runtime_error(msg),
				error_code_(code)
			{
			}

			[[nodiscard]] int getErrorCode() const { return error_code_; }
	};



	struct BatchContainer
	{
			llama_batch batch;

			explicit BatchContainer(const llama_batch &batch) :
				batch(batch)
			{
			}
			~BatchContainer() { llama_batch_free(batch); }

			BatchContainer(const BatchContainer &)			  = delete;
			BatchContainer &operator=(const BatchContainer &) = delete;
	};



	Call::Call(const boost::uuids::uuid) {}

	Call::Call(UuidGenerator &generator) :
		id(generator.generate())
	{
	}


	llama_pos Call::getPosition() const { return position_; }

	llama_seq_id Call::getSequenceId() const { return sequenceId_; }

	bool Call::initialized() const { return position_ > 0; }

	bool Call::assigned() const { return sequenceId_ != -1; }

	void Call::assign(const uint32_t sequenceId) { sequenceId_ = sequenceId; }

	void Call::restart() { position_ = 0; }

	void Call::incrementPosition() { position_++; }

	void Call::movePosition(const llama_pos change) { position_ += change; }

	std::string Call::info() const
	{
		fmt::memory_buffer buffer;

		fmt::format_to(
			std::back_inserter(buffer),
			"ID : {}, sequenceID : {}, tokenPosition : {}",
			to_string(id),
			sequenceId_,
			position_
		);

		return fmt::to_string(buffer);
	}



	std::string Parameters::print() const
	{
		fmt::memory_buffer buffer;

		fmt::format_to(
			std::back_inserter(buffer),
			"InferenceParameters{{modelFile=\"{}\", maxParallelSequences={}, "
			"maxTotalBatchSize={}, batchSize={}, totalContextSize={}, "
			"nCPUThreads={}, "
			"nGPUStoredLayers={}, decodingStrategy={}}}",
			modelFile != nullptr ? modelFile : "(null)",
			maxParallelSequences,
			maxTotalBatchSize,
			batchSize,
			totalContextSize,
			nCPUThreads,
			nGPUStoredLayers,
			static_cast<int>(decodingStrategy)
		);

		return fmt::to_string(buffer);
	}



	class ModelSession
	{
		public:

			explicit ModelSession(const Parameters &params);
			~ModelSession();

			ModelSession(const ModelSession &)			  = delete;
			ModelSession &operator=(const ModelSession &) = delete;

			static llama_model		 *loadModel(const Parameters &params);
			static const llama_vocab *loadVocabulary(const llama_model *model);
			static llama_context	 *startContext(const Parameters &params, llama_model *model);
			static llama_sampler	 *initSampler(const Parameters &params);

			void run(Call &call);
			void clearMemory() const;

		private:

			llama_model		  *model_	= nullptr;
			const llama_vocab *vocab_	= nullptr;
			llama_context	  *ctx_		= nullptr;
			llama_sampler	  *sampler_ = nullptr;

			Parameters				 parameters_;
			std::vector<llama_token> tokens_;

			void					  tokenize(const std::string &text);
			void					  generate(const llama_batch &batch) const;
			void					  prefill(llama_batch &batch, Call &call) const;
			[[nodiscard]] llama_token sample(std::string *output, bool consolePrint) const;
			void					  decode(llama_batch &batch, Call &call) const;
	};


	llama_model *ModelSession::loadModel(const Parameters &params)
	{
		LOG_DEBUG("loading model from \"{}\"", params.modelFile);

		llama_model_params model_params = llama_model_default_params();

		model_params.n_gpu_layers = params.nGPUStoredLayers;

		llama_model *model = llama_model_load_from_file(params.modelFile, model_params);

		if (model == nullptr)
		{
			LOG_ERROR("cannot load model from \"{}\"", params.modelFile);
			throw std::runtime_error("failed to load model from file");
		}

		LOG_DEBUG("model loaded from \"{}\"", params.modelFile);

		return model;
	}

	llama_context *ModelSession::startContext(const Parameters &params, llama_model *model)
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
			throw std::runtime_error("failed to initialize context");
		}

		LOG_DEBUG("inference context started");

		return ctx;
	}

	llama_sampler *ModelSession::initSampler(const Parameters &params)
	{
		LOG_DEBUG(
			"initializing sampler for \"{}\" decoding strategy",
			static_cast<int>(params.decodingStrategy)
		);

		llama_sampler *sampler = nullptr;

		switch (params.decodingStrategy)
		{
		case DecodingStrategy::Greedy:
			sampler = llama_sampler_init_greedy();
			break;
		}

		if (sampler == nullptr)
		{
			LOG_DEBUG(
				"failed to initialize sampler for \"{}\" decoding strategy",
				static_cast<int>(params.decodingStrategy)
			);
			throw std::runtime_error("failed to initialize sampler");
		}

		LOG_DEBUG(
			"sampler initialized for \"{}\" decoding strategy",
			static_cast<int>(params.decodingStrategy)
		);

		return sampler;
	}

	const llama_vocab *ModelSession::loadVocabulary(const llama_model *model)
	{
		LOG_DEBUG("loading model vocabulary");

		if (const llama_vocab *vocab = llama_model_get_vocab(model); vocab == nullptr)
		{
			LOG_ERROR("cannot load model vocabulary");
			throw std::runtime_error("failed to load vocabulary");
		}
		else
		{
			LOG_DEBUG("loaded model vocabulary");
			return vocab;
		}
	}


	ModelSession::ModelSession(const Parameters &params) :
		parameters_(params),
		tokens_(std::vector<llama_token>(400))
	{
		auto model = std::unique_ptr<llama_model, decltype(&llama_model_free)>(
			loadModel(params),
			&llama_model_free
		);

		const auto *vocab = loadVocabulary(model.get());

		auto ctx = std::unique_ptr<llama_context, decltype(&llama_free)>(
			startContext(params, model.get()),
			&llama_free
		);

		auto sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>(
			initSampler(params),
			&llama_sampler_free
		);

		model_	 = model.release();
		vocab_	 = vocab;
		ctx_	 = ctx.release();
		sampler_ = sampler.release();

		LOG_INFO("successfully created new `Inference` instance with:\n{}", params.print());
	}

	ModelSession::~ModelSession()
	{
		LOG_INFO("destructing `Inference` instance");

		llama_sampler_free(sampler_);
		llama_free(ctx_);
		llama_model_free(model_);
	}


	void ModelSession::tokenize(const std::string &text)
	{
		LOG_DEBUG("starting tokenization of \"{}\"", text);

		auto nTokens = llama_tokenize(
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
			// previous capacity was not enough
			tokens_.resize(-nTokens, 0);

			nTokens = llama_tokenize(
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
		}

		LOG_DEBUG("tokenization finished. {} tokens were prepared", nTokens);

		// removing redundant elements
		tokens_.resize(nTokens);
	}

	void ModelSession::generate(const llama_batch &batch) const
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

	void ModelSession::prefill(llama_batch &batch, Call &call) const
	{
		LOG_DEBUG("{} :: running prefill stage for {} tokens", call.info(), tokens_.size());

		size_t	   offset		  = 0;
		const auto chunk_capacity = static_cast<size_t>(parameters_.batchSize);

		while (offset < tokens_.size())
		{
			const auto chunk_size = std::min(chunk_capacity, tokens_.size() - offset);

			for (size_t i = 0; i < chunk_size; i++)
			{
				batch.logits[i]	   = false;
				batch.token[i]	   = tokens_[offset + i];
				batch.pos[i]	   = call.getPosition() + static_cast<llama_pos>(i);
				batch.n_seq_id[i]  = 1;
				batch.seq_id[i][0] = call.getSequenceId();
			}

			batch.logits[chunk_size - 1] = (offset + chunk_size == tokens_.size());
			batch.n_tokens				 = static_cast<int32_t>(chunk_size);

			generate(batch);

			call.movePosition(batch.n_tokens);

			offset += chunk_size;
		}

		LOG_DEBUG("{} :: prefill stage completed", call.info());
	}

	llama_token ModelSession::sample(std::string *output, const bool consolePrint) const
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

		const auto nBytes =
			llama_token_to_piece(vocab_, newTokenId, buffer, sizeof(buffer), 0, false);

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

	void ModelSession::decode(llama_batch &batch, Call &call) const
	{
		LOG_DEBUG("{} :: running decode stage", call.info());

		call.output.clear();

		for (auto tokenId = sample(&call.output, true); !llama_vocab_is_eog(vocab_, tokenId);
			 tokenId	  = sample(&call.output, true))
		{
			batch.n_tokens	   = 1;
			batch.token[0]	   = tokenId;
			batch.pos[0]	   = call.getPosition();
			batch.n_seq_id[0]  = 1;
			batch.seq_id[0][0] = call.getSequenceId();
			batch.logits[0]	   = true;

			generate(batch);

			call.incrementPosition();
		}

		LOG_DEBUG("{} :: generation finished: \"{}\"", call.info(), call.output);
	}

	void ModelSession::run(Call &call)
	{
		if (!call.assigned())
		{
			// TODO
			call.assign(0);
		}

		// TODO: add attaching to template
		if (!call.input.prompt().empty())
		{
			tokenize(call.input.prompt());

			if (!tokens_.empty())
			{
				const auto prefillBatchSize =
					std::max<size_t>(1, std::min(tokens_.size(), parameters_.batchSize));

				auto prefillBatch = BatchContainer(
					llama_batch_init(prefillBatchSize, 0, parameters_.maxParallelSequences)
				);

				prefill(prefillBatch.batch, call);
			}
		}

		auto decodeBatch = BatchContainer(llama_batch_init(1, 0, 1));

		decode(decodeBatch.batch, call);
	}

	void ModelSession::clearMemory() const { llama_memory_clear(llama_get_memory(ctx_), false); }



	using Sessions = std::unordered_map<boost::uuids::uuid, ModelSession>;



	struct Engine::impl
	{
			Sessions sessions_;
	};

	Engine::Engine() :
		impl_(std::make_unique<impl>())
	{
		llama_backend_init();
	}

	Engine::~Engine() { llama_backend_free(); }


	boost::uuids::uuid Engine::newSession(const Parameters &params, UuidGenerator &generator) const
	{
		while (true)
		{
			const auto uuid4 = generator.generate();

			if (auto [_, inserted] = impl_->sessions_.emplace(uuid4, params); inserted)
			{
				return uuid4;
			}
		}
	}

	void Engine::run(const boost::uuids::uuid &sessionId, Call &call) const
	{
		impl_->sessions_.at(sessionId).run(call);
	}
} // namespace Inference
