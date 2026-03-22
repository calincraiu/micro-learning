#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

#include "util/data/Dataset.h"

#include "core/Tape.h"
#include "core/Value.h"

/// <summary>
/// Data sample with a feature vector and scalar target.
/// </summary>
struct Sample {
	std::vector<Value> X;
	Value y;
};

class Dataloader {
public:
    Dataloader(Dataset& dataset, Tape* tape, bool shuffle = false);

    /// <summary>
    /// Gets entire dataset worth of samples. Must be called at the start of every epoch.
    /// </summary>
    /// <returns>one full epoch samples.</returns>
    std::vector<Sample> get_epoch_samples();

private:
    Dataset& m_dataset;
    Tape* m_tape;
    bool m_shuffle;
    std::vector<size_t> m_indices; // shuffling indices

    void shuffle_indices();
};

#endif // DATALOADER_H	