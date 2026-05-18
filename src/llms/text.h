#pragma once


#include <string>
#include <string_view>



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

		[[nodiscard]] bool				 validated() const;
		[[nodiscard]] const std::string &prompt() const;

	protected:

		void initialize(std::string &text);

		virtual void validate(std::string &text)  = 0;
		virtual void sanitize(std::string &text)  = 0;
		virtual void normalize(std::string &text) = 0;

		std::string prompt_;

	private:

		bool validated_ = false;
};



class UserPrompt : public PromptValidator
{
	public:

		UserPrompt() = default;
		UserPrompt(std::string text, const std::string_view &template_);

		static UserPrompt qwenPrompt(const std::string& text);

	protected:

		void validate(std::string &text) override;
		void sanitize(std::string &text) override;
		void normalize(std::string &text) override;

		void applyTemplate(const std::string_view &template_);
};
