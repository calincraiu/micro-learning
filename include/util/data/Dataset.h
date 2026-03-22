#ifndef DATASET_H
#define DATASET_H

#include <vector>

class Dataset {
public:
    /// <summary>
    /// Tabluar in-memory dataset. 
    /// </summary>
    /// <param name="features">N samples × M features.</param>
    /// <param name="targets">N scalar targets.</param>
    Dataset(const std::vector<std::vector<float>>& features,
        const std::vector<float>& targets);

    size_t size() const { return m_targets.size(); }

    // __getitem__
    std::pair<std::vector<float>, float> operator[](size_t idx) const;

    /// <summary>
    /// Factory function – creates Dataset from CSV file.
    /// Assumptions:
    ///     - First row = optional header (skipped if has_header == true);
    ///     - All other rows: features (floats), last column = target (float);
    ///     - No quoted fields containing commas;
    ///     - Consistent number of columns;
    /// </summary>
    /// <param name="filename">path to the CSV file.</param>
    /// <param name="has_header">whether the CSV has a header or not.</param>
    /// <param name="delimiter">char delimiter - e.g. a comma - ','</param>
    /// <returns>a Dataset object.</returns>
    static Dataset from_csv(const std::string& filename, bool has_header, char delimiter);

private:
    std::vector<std::vector<float>> m_features;
    std::vector<float> m_targets;
};



#endif // DATASET_H