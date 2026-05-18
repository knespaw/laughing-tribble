#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>

#include "utils/logger.h"

#include "text.h"



constexpr std::string_view QWEN_PROMPT_TEMPLATE =
	"<|im_start|>user\n<USR_PROMPT><|im_end|>\n<|im_start|>assistant\n<think></"
	"think>";


constexpr std::string_view USR_PROMPT_PLACEHOLDER = "<USR_PROMPT>";


constexpr size_t MAX_USER_INPUT_LENGTH = 1000;



namespace TextRules
{
	[[nodiscard]] std::string asciiLower(const std::string_view text)
	{
		std::string lowered;
		lowered.reserve(text.size());

		std::ranges::transform(
			text,
			lowered.begin(),
			[](const unsigned char c) { return static_cast<char>(std::tolower(c)); }
		);

		return lowered;
	}

	void ensureMaxLength(const std::string_view text)
	{
		if (text.size() > MAX_USER_INPUT_LENGTH)
		{
			LOG_ERROR(
				"user input exceeds maximum allowed length: {} > {}",
				text.size(),
				MAX_USER_INPUT_LENGTH
			);
			throw std::invalid_argument("user input exceeds maximum allowed length");
		}
	}

	void ensureNoNulBytes(const std::string_view text)
	{
		if (text.find('\0') != std::string_view::npos)
		{
			LOG_ERROR("user input contains NUL byte");
			throw std::invalid_argument("user input contains NUL byte");
		}
	}

	[[nodiscard]] bool isValidUtf8(const std::string_view text)
	{
		size_t i = 0;

		while (i < text.size())
		{
			const auto ch = static_cast<unsigned char>(text[i]);

			if ((ch & 0x80) == 0)
			{
				i++;
				continue;
			}

			size_t	 continuation_count = 0;
			uint32_t code_point			= 0;

			if ((ch & 0xE0) == 0xC0)
			{
				continuation_count = 1;
				code_point		   = ch & 0x1F;
				if (code_point == 0)
				{
					return false;
				}
			}
			else if ((ch & 0xF0) == 0xE0)
			{
				continuation_count = 2;
				code_point		   = ch & 0x0F;
			}
			else if ((ch & 0xF8) == 0xF0)
			{
				continuation_count = 3;
				code_point		   = ch & 0x07;
				if (code_point > 0x04)
				{
					return false;
				}
			}
			else
			{
				return false;
			}

			if (i + continuation_count >= text.size())
			{
				return false;
			}

			for (size_t j = 1; j <= continuation_count; j++)
			{
				const auto continuation = static_cast<unsigned char>(text[i + j]);
				if ((continuation & 0xC0) != 0x80)
				{
					return false;
				}

				code_point = code_point << 6 | continuation & 0x3F;
			}

			if ((continuation_count == 1 && code_point < 0x80) ||
				(continuation_count == 2 && code_point < 0x800) ||
				(continuation_count == 3 && code_point < 0x10000) || code_point > 0x10FFFF ||
				(code_point >= 0xD800 && code_point <= 0xDFFF))
			{
				return false;
			}

			i += continuation_count + 1;
		}

		return true;
	}

	void ensureValidUtf8(const std::string_view text)
	{
		if (!isValidUtf8(text))
		{
			LOG_ERROR("user input is not valid UTF-8");
			throw std::invalid_argument("user input is not valid UTF-8");
		}
	}

	void ensureNoPromptInjection(const std::string_view text)
	{
		static constexpr std::array<std::string_view, 12> blocked_patterns = {
			"ignore previous instructions",
			"ignore all previous instructions",
			"disregard previous instructions",
			"system prompt",
			"developer message",
			"reveal your instructions",
			"reveal the hidden prompt",
			"show me your prompt",
			"<|im_start|>",
			"<|im_end|>",
			"<|system|>",
			"<|assistant|>",
		}; // TODO

		const auto lowered = asciiLower(text);

		for (const auto pattern : blocked_patterns)
		{
			if (text.find(pattern) != std::string::npos)
			{
				LOG_ERROR("potential prompt injection detected due to pattern \"{}\"", pattern);
				throw std::invalid_argument("potential prompt injection detected");
			}
		}
	}


	void stripHtmlTags(std::string &text)
	{
		static const std::regex scriptOrStyle(
			R"(<\s*(script|style)\b[^>]*>[\s\S]*?<\s*/\s*\1\s*>)",
			std::regex::icase
		);
		static const std::regex tags(R"(<[^>]+>)");

		text = std::regex_replace(text, scriptOrStyle, " ");
		text = std::regex_replace(text, tags, " ");
	}

	void stripMarkdownCodeFences(std::string &text)
	{
		static const std::regex fencedCode(R"(```[\s\S]*?```)");
		static const std::regex inlineCode(R"(`[^`\n]+`)");

		text = std::regex_replace(text, fencedCode, " ");
		text = std::regex_replace(text, inlineCode, " ");
	}

	void stripJavascriptElements(std::string &text)
	{
		static const std::regex jsFragments(
			R"(\b(?:javascript\s*:|eval\s*\(|function\s*\(|settimeout\s*\(|setinterval\s*\(|document\s*\.\s*cookie|window\s*\.\s*location|on[a-z]+\s*=))",
			std::regex::icase
		);

		text = std::regex_replace(text, jsFragments, " ");
	}

	void stripSqlElements(std::string &text)
	{
		static const std::regex sqlLineComments(R"(--[^\n\r]*)");
		static const std::regex sqlBlockComments(R"(/\*[\s\S]*?\*/)");
		static const std::regex sqlStatements(
			R"(\b(?:select|insert|update|delete|drop|alter|create|truncate|replace|exec|execute)\b[^;\n\r]*;?)",
			std::regex::icase
		);

		text = std::regex_replace(text, sqlLineComments, " ");
		text = std::regex_replace(text, sqlBlockComments, " ");
		text = std::regex_replace(text, sqlStatements, " ");
	}

	void stripControlCharacters(std::string &text)
	{
		std::erase_if(
			text,
			[](const unsigned char ch) { return std::iscntrl(ch) && ch != '\n' && ch != '\t'; }
		);
	}


	void normalizeLineEndings(std::string &text)
	{
		size_t pos = 0;
		while ((pos = text.find("\r\n", pos)) != std::string::npos)
		{
			text.replace(pos, 2, "\n");
		}

		std::ranges::replace(text, '\r', '\n');
	}

	void normalizeWhitespace(std::string &text)
	{
		std::string normalized;
		normalized.reserve(text.size());

		bool in_space	= false;
		bool in_newline = false;

		for (const unsigned char ch : text)
		{
			if (ch == '\n')
			{
				if (!normalized.empty() && normalized.back() != '\n')
				{
					normalized.push_back('\n');
				}
				in_space   = false;
				in_newline = true;
				continue;
			}

			if (std::isspace(ch))
			{
				if (!normalized.empty() && !in_space && !in_newline)
				{
					normalized.push_back(' ');
				}
				in_space = true;
				continue;
			}

			normalized.push_back(static_cast<char>(ch));
			in_space   = false;
			in_newline = false;
		}

		const auto begin = normalized.find_first_not_of(" \n\t");
		if (begin == std::string::npos)
		{
			text.clear();
			return;
		}

		const auto end = normalized.find_last_not_of(" \n\t");
		text		   = normalized.substr(begin, end - begin + 1);
	}
} // namespace TextRules


bool PromptValidator::validated() const { return validated_; }

const std::string &PromptValidator::prompt() const
{
	if (!validated_)
	{
		LOG_ERROR("prompt has not been validated");
		throw std::runtime_error("prompt has not been validated");
	}

	return prompt_;
}

void PromptValidator::initialize(std::string &text)
{
	validate(text);
	sanitize(text);
	normalize(text);

	validated_ = true;
	prompt_	   = text;
}



UserPrompt::UserPrompt(std::string text, const std::string_view &template_)
{
	initialize(text);
	applyTemplate(template_);
}


UserPrompt UserPrompt::qwenPrompt(const std::string& text) {
	return {text, QWEN_PROMPT_TEMPLATE};
}


void UserPrompt::validate(std::string &text)
{
	LOG_DEBUG("validating prompt input");

	TextRules::ensureNoNulBytes(text);
	TextRules::ensureMaxLength(text);
	TextRules::ensureValidUtf8(text);
	TextRules::ensureNoPromptInjection(text);
}

void UserPrompt::sanitize(std::string &text)
{
	LOG_DEBUG("sanitizing prompt input");

	TextRules::stripHtmlTags(text);
	TextRules::stripMarkdownCodeFences(text);
	TextRules::stripJavascriptElements(text);
	TextRules::stripSqlElements(text);
	TextRules::stripControlCharacters(text);
}

void UserPrompt::normalize(std::string &text)
{
	LOG_DEBUG("normalizing prompt input");

	TextRules::normalizeLineEndings(text);
	TextRules::normalizeWhitespace(text);
}

void UserPrompt::applyTemplate(const std::string_view &template_)
{
	const auto idx = template_.find(USR_PROMPT_PLACEHOLDER);

	if (idx == std::string::npos)
	{
		LOG_ERROR(
			"`invalid prompt template \"{}\". should contain \"{}\" inside",
			template_,
			USR_PROMPT_PLACEHOLDER
		);
		throw std::invalid_argument("invalid prompt template");
	}

	std::string templated_text;
	templated_text.reserve(template_.size() - USR_PROMPT_PLACEHOLDER.size() + prompt_.size());

	templated_text.append(template_, 0, idx);
	templated_text.append(prompt_);
	templated_text.append(template_, idx + USR_PROMPT_PLACEHOLDER.size(), std::string::npos);

	prompt_ = std::move(templated_text);
}
