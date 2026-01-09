#pragma once

#include <string>
#include <vector>

struct SystemMetrics {
    int step = 0;
    int dna_pool_size = 0;
    float avg_agent_energy = 0.0f;
};

struct ReportOptions {
    std::string dump_dir;
    std::string dump_prefix;
    std::string report_html_path;
    int downsample = 32;
    bool paper_mode = false;
    bool global_normalization = false;
    int hist_bins = 64;
    bool include_sparklines = true;
    std::string scenario_summary;
    std::vector<SystemMetrics> system_metrics;
};

bool generate_dump_report_html(const ReportOptions &opts, std::string &error);
