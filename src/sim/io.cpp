#include "io.h"

#include <fstream>
#include <sstream>

namespace {
bool parse_line(const std::string &line, std::vector<float> &row) {
    row.clear();
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        if (cell.empty()) {
            continue;
        }
        try {
            row.push_back(std::stof(cell));
        } catch (...) {
            return false;
        }
    }
    return !row.empty();
}
} // namespace

bool load_grid_csv(const std::string &path, GridData &out, std::string &error) {
    std::ifstream file(path);
    if (!file.is_open()) {
        error = "Datei konnte nicht geoeffnet werden: " + path;
        return false;
    }

    std::vector<std::vector<float>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.size() >= 1 && line[0] == '#') {
            continue;
        }
        std::vector<float> row;
        if (!parse_line(line, row)) {
            error = "Ungueltige CSV-Zeile: " + line;
            return false;
        }
        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        error = "CSV-Datei ist leer: " + path;
        return false;
    }

    const int width = static_cast<int>(rows.front().size());
    const int height = static_cast<int>(rows.size());
    for (const auto &row : rows) {
        if (static_cast<int>(row.size()) != width) {
            error = "Inkonsistente Zeilenlaengen in CSV: " + path;
            return false;
        }
    }

    out.width = width;
    out.height = height;
    out.values.clear();
    out.values.reserve(width * height);
    for (const auto &row : rows) {
        out.values.insert(out.values.end(), row.begin(), row.end());
    }
    return true;
}
