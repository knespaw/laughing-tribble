#pragma once


#include <string>

#include "llama.h"

#include "text.h"
#include "utils.h"



namespace Inference
{
	enum class DecodingStrategy
	{
		Greedy = 0,
	};



	struct Parameters
	{
			const char		*modelFile;
			uint32_t		 maxParallelSequences;
			uint32_t		 maxTotalBatchSize;
			size_t			 batchSize;
			uint32_t		 totalContextSize;
			int32_t			 nCPUThreads;
			int32_t			 nGPUStoredLayers;
			DecodingStrategy decodingStrategy;

			[[nodiscard]] std::string print() const;
	};



	class Call
	{
		public:

			explicit Call(UuidGenerator &generator);
			explicit Call(boost::uuids::uuid);

			const boost::uuids::uuid id;
			UserPrompt				 input;
			std::string				 output;

			[[nodiscard]] llama_pos	   getPosition() const;
			[[nodiscard]] llama_seq_id getSequenceId() const;
			[[nodiscard]] std::string  info() const;
			[[nodiscard]] bool		   initialized() const;
			[[nodiscard]] bool		   assigned() const;
			void					   assign(uint32_t sequenceId);
			void					   restart();
			void					   incrementPosition();
			void					   movePosition(llama_pos change);

		private:

			llama_seq_id sequenceId_ = -1;
			llama_pos	 position_	 = 0;
	};



	class Engine
	{
		public:

			Engine();
			~Engine();

			Engine(const Engine &)			  = delete;
			Engine &operator=(const Engine &) = delete;
			Engine(Engine &&)				  = delete;
			Engine &operator=(Engine &&)	  = delete;

			boost::uuids::uuid newSession(const Parameters &params, UuidGenerator &generator) const;
			void			   run(const boost::uuids::uuid &sessionId, Call &call) const;

		private:

			struct impl;
			std::unique_ptr<impl> impl_;
	};
} // namespace Inference
