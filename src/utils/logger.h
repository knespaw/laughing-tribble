#pragma once


#include <memory>

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"



// clang-format off
#define LOG_TRACE(...) spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::trace, __VA_ARGS__)
#define LOG_DEBUG(...) spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::debug, __VA_ARGS__)
#define LOG_INFO(...) spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::info, __VA_ARGS__)
#define LOG_WARN(...) spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::warn, __VA_ARGS__)
#define LOG_ERROR(...) spdlog::log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::err, __VA_ARGS__)
// clang-format on



namespace Logger
{
	inline void init()
	{
		auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

		// TODO: change it later
		auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
			std::string(LOG_DIR) + "/temp.log",
			true
		);

		const auto logger = std::make_shared<spdlog::logger>(
			"xprv",
			spdlog::sinks_init_list{console_sink, file_sink}
		);

		logger->set_pattern(
			"[%Y-%m-%d %H:%M:%S.%e] :::: [P:%P] :::: [T:%t] :::: "
			"[%s:%# -> %!] :::: [%l] :::: %v"
		);

		switch (LOG_LEVEL)
		{
		case 0:
			logger->set_level(spdlog::level::trace);
			break;
		case 1:
			logger->set_level(spdlog::level::debug);
			break;
		case 2:
			logger->set_level(spdlog::level::info);
			break;
		case 3:
			logger->set_level(spdlog::level::warn);
			break;
		default:
			logger->set_level(spdlog::level::err);
		}

		// logs are flushed only for INFO level
		logger->flush_on(spdlog::level::info);

		spdlog::set_default_logger(logger);
	}
} // namespace Logger
