#if defined(_WIN32)
#include <windows.h>
#endif

#include "common.h"
#include "config.h"
#include <filesystem>
#include <iostream>
#include <log4cplus/log4cplus.h>
#include <onnxruntime_cxx_api.h>
#include <expected>



int main(int argc, char* argv[])
{
    log4cplus::initialize();
    if (!std::filesystem::exists("logs"))
    {
        std::filesystem::create_directory("logs");
    }
    log4cplus::PropertyConfigurator::doConfigure(
        LOG4CPLUS_STRING_TO_TSTRING("config/log4cplus.properties"));
    log4cplus::Logger logger =
        log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("xmake-template-cpp"));
    LOG4CPLUS_DEBUG(logger, "This is a demo");
    LOG4CPLUS_INFO(logger, LOG4CPLUS_STRING_TO_TSTRING(
                       std::format("GIT_COMMIT:{}", GIT_COMMIT)));
    std::cout << "main" << std::endl;
    std::cout << "a+b=" << common::add(1, 2) << std::endl;

    const auto& info = common::AppBuildInfo::instance();
    std::cout << info.toString();
};
