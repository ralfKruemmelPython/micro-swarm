#pragma once

#include <string>

struct ReportOptions {
    std::string dump_dir;
    std::string dump_prefix;
    std::string report_html_path;
    int downsample = 32;
};

bool generate_dump_report_html(const ReportOptions &opts, std::string &error);
