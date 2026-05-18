#include <string>

#include "utils.h"
#include "utils/logger.h"

#include "llms.h"



int main()
{
	Logger::init();

	auto idGenerator = UuidGenerator();

	auto inferenceEngine = Inference::Engine();

	constexpr auto inferParams = Inference::Parameters{
		QWEN_FILE,
		1,
		512,
		512,
		4096,
		8,
		-1,
		Inference::DecodingStrategy::Greedy
	};

	const auto sessionId = inferenceEngine.newSession(inferParams, idGenerator);

	auto call = Inference::Call(idGenerator);

	call.input = UserPrompt::qwenPrompt("What is the capital of Poland?");

	inferenceEngine.run(sessionId, call);

	call.input = UserPrompt::qwenPrompt("What is population of this city?");

	inferenceEngine.run(sessionId, call);

	call.input = UserPrompt::qwenPrompt("When the city was founded?");

	inferenceEngine.run(sessionId, call);

	return 0;
}
