#include <iostream>
#include <string>
#include <vector>

#include "llama.h"

#include "utils/logger.h"

#include "llms.h"




auto MODEL_PATH = "../model_files/Qwen3-8B-Q5_K_M.gguf";



void setup()
{
	std::cout << "Initializing Metal Backend..." << std::endl;

	llama_backend_init();

	std::cout << "\n=== System Info ===" << std::endl;
	std::cout << llama_print_system_info() << std::endl;

	Logger::init();
}


llama_model *load_model()
{
	std::cout << "Loading Model..." << std::endl;

	llama_model_params model_params = llama_model_default_params();

	model_params.n_gpu_layers = -1;

	llama_model *model = llama_model_load_from_file(MODEL_PATH, model_params);

	return model;
}


llama_context *create_context(llama_model *model)
{
	std::cout << "Creating context..." << std::endl;

	llama_context_params ctx_params = llama_context_default_params();

	ctx_params.n_ctx = 4096;

	ctx_params.n_threads = 8;

	std::cout << "Allocating context (KV cache)..." << std::endl;

	llama_context *ctx = llama_init_from_model(model, ctx_params);

	return ctx;
}


void cleanup(llama_model *model, llama_context *ctx)
{
	llama_free(ctx);
	llama_model_free(model);
	llama_backend_free();
}



int main()
{
	llama_backend_init();

	auto params = InferenceParameters {
		QWEN_FILE,
		1,
		512,
		512,
		4096,
		8,
		-1,
		GREEDY,
		"<|im_start|>user\n<PROMPT><|im_end|>\n<|im_start|>assistant\n"
	};

	{
		auto inference = Inference(params);
		inference.infer("What is the capital of Poland?");
		inference.infer("What is the capital of France?");
	}

	llama_backend_free();

	return 0;
}
