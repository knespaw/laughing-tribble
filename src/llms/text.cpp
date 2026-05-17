#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "utils/logger.h"



constexpr std::string_view QWEN_PROMPT_TEMPLATE = "<|im_start|>user\n<USR_PROMPT><|im_end|>\n<|im_start|>assistant\n<think></think>";


constexpr std::string_view USR_PROMPT_PLACEHOLDER = "<USR_PROMPT>";


constexpr size_t MAX_USER_INPUT_LENGTH = 1000;



/**
 * @class PromptValidator
 * @brief Blueprint for other classes which represent a validated and sanitized
 * textual input to a language model. It is enforced by using Non-Virtual
 * Interface (NVI) pattern.
 */
class PromptValidator
{
	public:

		virtual ~PromptValidator() = default;

		[[nodiscard]] bool validated() const { return validated_; }
		[[nodiscard]] virtual const std::string &prompt() const
		{
			if (!validated_)
			{
				throw std::runtime_error("prompt has not been validated");
			}

			return prompt_;
		}

	protected:

		virtual void initialize(std::string &text)
		{
			validate(text);
			sanitize(text);
			normalize(text);

			validated_ = true;
			prompt_	   = text;
		}

		virtual void validate(std::string &text)  = 0;
		virtual void sanitize(std::string &text)  = 0;
		virtual void normalize(std::string &text) = 0;

		std::string prompt_;

	private:

		bool validated_ = false;
};



/**
 * @class PromptProcessor
 * @brief Blueprint for other classes which represent a processed textual input
 * that can be directly fed to a language model. Subclass of @PromptValidator.
 */
class PromptProcessor : public PromptValidator
{
	public:

		~PromptProcessor() override = default;

		[[nodiscard]] bool processed() const { return processed_; }
		[[nodiscard]] const std::string &prompt() const override
		{
			if (!processed_)
			{
				throw std::runtime_error("prompt has not been processed");
			}

			return PromptValidator::prompt();
		}

	protected:

		void initialize(std::string &text) override
		{
			PromptValidator::initialize(text);

			process();

			processed_ = true;
		}

		void applyTemplate(const std::string_view &template_);

		virtual void process() = 0;

	private:

		bool processed_ = false;
};


void PromptProcessor::applyTemplate(const std::string_view &template_) {
	const auto idx = template_.find(USR_PROMPT_PLACEHOLDER);

	if (idx == std::string::npos)
	{
		LOG_ERROR(
			"`Inference` initialization failed. invalid prompt template "
			"\"{}\". should contain \"{}\" inside",
			template_,
			USR_PROMPT_PLACEHOLDER
		);
		throw std::invalid_argument("invalid prompt template");
	}

	std::string templated_text;
	templated_text.reserve(
		template_.size() - USR_PROMPT_PLACEHOLDER.size() + prompt_.size()
	);

	templated_text.append(template_, 0, idx);
	templated_text.append(prompt_);
	templated_text.append(
		template_,
		idx + USR_PROMPT_PLACEHOLDER.size(),
		std::string::npos
	);

	prompt_ = std::move(templated_text);
}



class QwenPrompt : public PromptProcessor {
	public:
		explicit QwenPrompt(std::string text) { PromptProcessor::initialize(text); }

	protected:
		void validate(std::string &text) override;
		void sanitize(std::string &text) override;
		void normalize(std::string &text) override;
		void process() override;
};


void QwenPrompt::process() {
	applyTemplate(QWEN_PROMPT_TEMPLATE);
}
