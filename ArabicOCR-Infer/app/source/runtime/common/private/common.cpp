#include "../common.h"
#include "config.h"
#include <format>
#include <string>


namespace common {

class AppBuildInfo::Impl {
public:
  Impl()
      : m_version(VERSION), m_versionMinor(VERSION_MINOR),
        m_versionAlter(VERSION_ALTER), m_versionBuild(VERSION_BUILD),
        m_plat(PLAT), m_arch(ARCH), m_mode(MODE), m_debug(DEBUG), m_os(OS),
        m_gitCommit(GIT_COMMIT), m_gitCommitLong(GIT_COMMIT_LONG),
        m_gitCommitDate(GIT_COMMIT_DATE), m_gitBranch(GIT_BRANCH),
        m_gitTag(GIT_TAG), m_gitTagLong(GIT_TAG_LONG), m_gitCustom(GIT_CUSTOM),
        m_projectName(PROJECT_NAME) {}

  const char *m_version;
  const char *m_versionMinor;
  const char *m_versionAlter;
  const char *m_versionBuild;
  const char *m_plat;
  const char *m_arch;
  const char *m_mode;
  const char *m_debug;
  const char *m_os;
  const char *m_gitCommit;
  const char *m_gitCommitLong;
  const char *m_gitCommitDate;
  const char *m_gitBranch;
  const char *m_gitTag;
  const char *m_gitTagLong;
  const char *m_gitCustom;
  const char *m_projectName;
};

AppBuildInfo &AppBuildInfo::instance() {
  static AppBuildInfo instance;
  return instance;
}

AppBuildInfo::AppBuildInfo() : d_ptr(new Impl()) {}

AppBuildInfo::~AppBuildInfo() { delete d_ptr; }

const char *AppBuildInfo::version() const { return d_ptr->m_version; }
const char *AppBuildInfo::versionMinor() const { return d_ptr->m_versionMinor; }
const char *AppBuildInfo::versionAlter() const { return d_ptr->m_versionAlter; }
const char *AppBuildInfo::versionBuild() const { return d_ptr->m_versionBuild; }
const char *AppBuildInfo::plat() const { return d_ptr->m_plat; }
const char *AppBuildInfo::arch() const { return d_ptr->m_arch; }
const char *AppBuildInfo::mode() const { return d_ptr->m_mode; }
const char *AppBuildInfo::debug() const { return d_ptr->m_debug; }
const char *AppBuildInfo::os() const { return d_ptr->m_os; }
const char *AppBuildInfo::gitCommit() const { return d_ptr->m_gitCommit; }
const char *AppBuildInfo::gitCommitLong() const {
  return d_ptr->m_gitCommitLong;
}
const char *AppBuildInfo::gitCommitDate() const {
  return d_ptr->m_gitCommitDate;
}
const char *AppBuildInfo::gitBranch() const { return d_ptr->m_gitBranch; }
const char *AppBuildInfo::gitTag() const { return d_ptr->m_gitTag; }
const char *AppBuildInfo::gitTagLong() const { return d_ptr->m_gitTagLong; }
const char *AppBuildInfo::gitCustom() const { return d_ptr->m_gitCustom; }

const char *AppBuildInfo::projectName() const { return d_ptr->m_projectName; }

std::string AppBuildInfo::toString() const {
  return std::format("App Build Info:\n"
                     "  Project Name: {}\n"
                     "  Version: {}\n"
                     "  Version Minor: {}\n"
                     "  Version Alter: {}\n"
                     "  Version Build: {}\n"
                     "  Platform: {}\n"
                     "  Architecture: {}\n"
                     "  Mode: {}\n"
                     "  Debug: {}\n"
                     "  OS: {}\n"
                     "  Git Commit: {}\n"
                     "  Git Commit Long: {}\n"
                     "  Git Commit Date: {}\n"
                     "  Git Branch: {}\n"
                     "  Git Tag: {}\n"
                     "  Git Tag Long: {}\n"
                     "  Git Custom: {}\n",
                     projectName(), version(), versionMinor(), versionAlter(),
                     versionBuild(), plat(), arch(), mode(), debug(), os(),
                     gitCommit(), gitCommitLong(), gitCommitDate(), gitBranch(),
                     gitTag(), gitTagLong(), gitCustom());
}

int add(int a, int b) { return a + b; }

} // namespace common