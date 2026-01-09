#include "report.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "io.h"

namespace {
struct FieldStats {
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
    float stddev = 0.0f;
    float nonzero_ratio = 0.0f;
};

bool parse_dump_filename(const std::string &filename, const std::string &prefix, int &step, std::string &field) {
    const std::string tag = prefix + "_step";
    if (filename.rfind(tag, 0) != 0) {
        return false;
    }

    size_t pos = tag.size();
    if (pos + 6 >= filename.size()) {
        return false;
    }
    int step_val = 0;
    for (size_t i = 0; i < 6; ++i) {
        char c = filename[pos + i];
        if (c < '0' || c > '9') {
            return false;
        }
        step_val = step_val * 10 + (c - '0');
    }
    pos += 6;
    if (filename[pos] != '_') {
        return false;
    }
    pos += 1;
    if (filename.size() <= pos + 4) {
        return false;
    }
    if (filename.substr(filename.size() - 4) != ".csv") {
        return false;
    }
    std::string field_name = filename.substr(pos, filename.size() - pos - 4);
    if (field_name != "resources" && field_name != "pheromone" && field_name != "molecules" && field_name != "mycel") {
        return false;
    }
    step = step_val;
    field = field_name;
    return true;
}

FieldStats compute_stats(const std::vector<float> &values) {
    FieldStats stats;
    if (values.empty()) {
        return stats;
    }
    stats.min = values.front();
    stats.max = values.front();
    double sum = 0.0;
    int nonzero = 0;
    for (float v : values) {
        if (v < stats.min) stats.min = v;
        if (v > stats.max) stats.max = v;
        sum += v;
        if (v > 1e-6f) nonzero++;
    }
    stats.mean = static_cast<float>(sum / static_cast<double>(values.size()));
    double var_sum = 0.0;
    for (float v : values) {
        double d = static_cast<double>(v) - stats.mean;
        var_sum += d * d;
    }
    stats.stddev = static_cast<float>(std::sqrt(var_sum / static_cast<double>(values.size())));
    stats.nonzero_ratio = static_cast<float>(nonzero) / static_cast<float>(values.size());
    return stats;
}

std::vector<float> downsample_grid(int width, int height, const std::vector<float> &values, int target) {
    if (target <= 0 || width <= 0 || height <= 0) {
        return {};
    }
    std::vector<float> out(static_cast<size_t>(target) * static_cast<size_t>(target), 0.0f);
    for (int ty = 0; ty < target; ++ty) {
        int y0 = static_cast<int>(std::floor(static_cast<double>(ty) * height / target));
        int y1 = static_cast<int>(std::floor(static_cast<double>(ty + 1) * height / target));
        if (y1 <= y0) y1 = std::min(height, y0 + 1);
        for (int tx = 0; tx < target; ++tx) {
            int x0 = static_cast<int>(std::floor(static_cast<double>(tx) * width / target));
            int x1 = static_cast<int>(std::floor(static_cast<double>(tx + 1) * width / target));
            if (x1 <= x0) x1 = std::min(width, x0 + 1);
            double sum = 0.0;
            int count = 0;
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    sum += values[static_cast<size_t>(y) * width + x];
                    count++;
                }
            }
            float avg = (count > 0) ? static_cast<float>(sum / count) : 0.0f;
            out[static_cast<size_t>(ty) * target + tx] = avg;
        }
    }
    return out;
}

std::string render_svg_heatmap(const std::vector<float> &values, int size, float min, float max) {
    if (values.empty() || size <= 0) {
        return "";
    }
    const int cell = 4;
    const int w = size * cell;
    const int h = size * cell;
    std::ostringstream ss;
    ss << "<svg width=\"" << w << "\" height=\"" << h << "\" viewBox=\"0 0 " << w << " " << h
       << "\" xmlns=\"http://www.w3.org/2000/svg\" shape-rendering=\"crispEdges\">";
    float range = max - min;
    if (range <= 0.0f) {
        range = 1.0f;
    }
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float v = values[static_cast<size_t>(y) * size + x];
            float norm = (v - min) / range;
            if (norm < 0.0f) norm = 0.0f;
            if (norm > 1.0f) norm = 1.0f;
            int shade = static_cast<int>(std::round(norm * 255.0f));
            ss << "<rect x=\"" << x * cell << "\" y=\"" << y * cell << "\" width=\"" << cell
               << "\" height=\"" << cell << "\" fill=\"rgb(" << shade << "," << shade << "," << shade
               << ")\"/>";
        }
    }
    ss << "</svg>";
    return ss.str();
}

std::string make_relative_link(const std::filesystem::path &from_dir, const std::filesystem::path &to_file) {
    std::error_code ec;
    std::filesystem::path rel = std::filesystem::relative(to_file, from_dir, ec);
    if (ec) {
        return to_file.filename().generic_string();
    }
    return rel.generic_string();
}
} // namespace

bool generate_dump_report_html(const ReportOptions &opts, std::string &error) {
    if (opts.dump_dir.empty()) {
        error = "Dump-Verzeichnis ist leer";
        return false;
    }
    if (opts.dump_prefix.empty()) {
        error = "Dump-Prefix ist leer";
        return false;
    }

    std::filesystem::path dump_dir = opts.dump_dir;
    if (!std::filesystem::exists(dump_dir)) {
        error = "Dump-Verzeichnis existiert nicht: " + opts.dump_dir;
        return false;
    }

    std::map<int, std::map<std::string, std::filesystem::path>> mapping;
    for (const auto &entry : std::filesystem::directory_iterator(dump_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        int step = 0;
        std::string field;
        if (!parse_dump_filename(name, opts.dump_prefix, step, field)) {
            continue;
        }
        mapping[step][field] = entry.path();
    }

    if (mapping.empty()) {
        error = "Keine Dump-Dateien gefunden";
        return false;
    }

    const std::vector<std::string> fields = {"resources", "pheromone", "molecules", "mycel"};

    struct StepData {
        int step = 0;
        int width = 0;
        int height = 0;
        std::map<std::string, GridData> grids;
        std::map<std::string, FieldStats> stats;
        std::map<std::string, std::string> previews;
        std::map<std::string, std::filesystem::path> paths;
    };

    std::vector<StepData> steps;
    steps.reserve(mapping.size());

    for (const auto &pair : mapping) {
        StepData data;
        data.step = pair.first;
        for (const auto &field : fields) {
            auto it = pair.second.find(field);
            if (it == pair.second.end()) {
                error = "Fehlendes Feld " + field + " fuer Step " + std::to_string(pair.first);
                return false;
            }
            data.paths[field] = it->second;
        }
        steps.push_back(std::move(data));
    }

    for (auto &step : steps) {
        int step_width = 0;
        int step_height = 0;
        for (const auto &field : fields) {
            GridData grid;
            std::string load_error;
            if (!load_grid_csv(step.paths[field].string(), grid, load_error)) {
                error = "CSV-Fehler: " + load_error;
                return false;
            }
            if (grid.values.empty()) {
                error = "Leere CSV: " + step.paths[field].string();
                return false;
            }
            if (step_width == 0 && step_height == 0) {
                step_width = grid.width;
                step_height = grid.height;
            } else if (grid.width != step_width || grid.height != step_height) {
                error = "Inkonsistente Rastergroesse in Step " + std::to_string(step.step);
                return false;
            }
            step.grids[field] = std::move(grid);
        }
        step.width = step_width;
        step.height = step_height;

        for (const auto &field : fields) {
            const auto &grid = step.grids[field];
            step.stats[field] = compute_stats(grid.values);
            if (opts.downsample > 0) {
                std::vector<float> down = downsample_grid(grid.width, grid.height, grid.values, opts.downsample);
                step.previews[field] = render_svg_heatmap(down, opts.downsample, step.stats[field].min, step.stats[field].max);
            }
        }
    }

    std::filesystem::path report_path;
    if (opts.report_html_path.empty()) {
        report_path = dump_dir / (opts.dump_prefix + "_report.html");
    } else {
        report_path = opts.report_html_path;
    }

    std::ofstream out(report_path);
    if (!out.is_open()) {
        error = "Report konnte nicht geschrieben werden: " + report_path.string();
        return false;
    }

    std::filesystem::path report_dir = report_path.parent_path();
    out << "<!doctype html>\n";
    out << "<html><head><meta charset=\"utf-8\">";
    out << "<title>Micro-Swarm Dump Report</title>";
    out << "<style>";
    out << "body{font-family:Arial,Helvetica,sans-serif;margin:20px;color:#222;}";
    out << "table{border-collapse:collapse;width:100%;margin:10px 0;}";
    out << "th,td{border:1px solid #ccc;padding:6px 8px;vertical-align:top;}";
    out << "th{background:#f2f2f2;text-align:left;}";
    out << ".meta{margin-bottom:16px;}";
    out << ".preview{margin-top:4px;}";
    out << "</style></head><body>";
    out << "<h1>Micro-Swarm Dump Report</h1>";
    out << "<div class=\"meta\">";
    out << "<div>dump_dir: " << opts.dump_dir << "</div>";
    out << "<div>prefix: " << opts.dump_prefix << "</div>";
    out << "<div>steps: " << steps.size() << "</div>";
    out << "</div>";

    for (const auto &step : steps) {
        out << "<h2>Step " << step.step << "</h2>";
        out << "<table>";
        out << "<tr><th>Field</th><th>CSV</th><th>Stats</th><th>Preview</th></tr>";
        for (const auto &field : fields) {
            const auto &stats = step.stats.at(field);
            std::string link = make_relative_link(report_dir, step.paths.at(field));
            out << "<tr>";
            out << "<td>" << field << "</td>";
            out << "<td><a href=\"" << link << "\">" << link << "</a></td>";
            out << "<td>";
            out << "min=" << std::fixed << std::setprecision(4) << stats.min << "<br>";
            out << "max=" << std::fixed << std::setprecision(4) << stats.max << "<br>";
            out << "mean=" << std::fixed << std::setprecision(4) << stats.mean << "<br>";
            out << "stddev=" << std::fixed << std::setprecision(4) << stats.stddev << "<br>";
            out << "nonzero_ratio=" << std::fixed << std::setprecision(4) << stats.nonzero_ratio;
            out << "</td>";
            out << "<td>";
            if (opts.downsample > 0) {
                out << "<div class=\"preview\">" << step.previews.at(field) << "</div>";
            } else {
                out << "-";
            }
            out << "</td>";
            out << "</tr>";
        }
        out << "</table>";
    }

    out << "</body></html>";
    return true;
}
