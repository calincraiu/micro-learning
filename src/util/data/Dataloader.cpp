#include "util/data/Dataloader.h"


Dataloader::Dataloader(Dataset& dataset, Tape* tape, bool shuffle)
    : m_dataset(dataset), m_tape(tape), m_shuffle(shuffle)
{
    m_indices.resize(dataset.size());
    std::iota(m_indices.begin(), m_indices.end(), 0);
}

void Dataloader::shuffle_indices() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(m_indices.begin(), m_indices.end(), gen);
}

std::vector<Sample> Dataloader::get_epoch_samples() {
    if (m_shuffle) shuffle_indices();

    std::vector<Sample> epoch_samples;
    epoch_samples.reserve(m_dataset.size());

    for (size_t i = 0; i < m_dataset.size(); ++i) {
        size_t idx = m_shuffle ? m_indices[i] : i;
        auto [feat_raw, targ_raw] = m_dataset[idx];

        // Create fresh Value leaves on the tape
        std::vector<Value> feat_values;
        feat_values.reserve(feat_raw.size());
        for (float f : feat_raw) {
            feat_values.emplace_back(m_tape->create_leaf(f), m_tape);
        }
        Value target_value(m_tape->create_leaf(targ_raw), m_tape);

        epoch_samples.push_back({ std::move(feat_values), std::move(target_value) });
    }
    return epoch_samples;
}