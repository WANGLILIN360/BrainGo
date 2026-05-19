//! Population-level activity readout.
//!
//! - Region-mean LFP estimator (mean `v_mem`)
//! - FFT spectral peak detection (design doc §8.2)
//! - Functional connectivity matrix via Pearson correlation (§8.2)

use crate::storage::mmap_db::BrainDB;

/// Return the mean membrane voltage across all neurons whose `region_id`
/// matches `region_id`.
///
/// The `dynamic_states` slice must come from a [`BrainDB`] / [`crate::sim::Simulation`]
/// whose neuron count matches `db`. Reading the slice avoids needing a
/// `&Simulation` here so this function works equally well on a freshly
/// loaded DB and on a running simulation.
pub fn region_mean_lfp(
    db: &BrainDB,
    region_id: u32,
    dynamic_states: &[crate::core::neuron::NeuronState],
) -> f32 {
    let attrs = db.neuron_attrs();
    debug_assert_eq!(attrs.len(), dynamic_states.len());

    let mut sum = 0.0_f64;
    let mut n = 0usize;
    for (a, s) in attrs.iter().zip(dynamic_states.iter()) {
        if a.region_id == region_id {
            sum += s.v_mem as f64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64) as f32
    }
}

// ── FFT oscillation detection ────────────────────────────────────────────

/// Spectral peak in a power spectrum.
#[derive(Clone, Debug)]
pub struct SpectralPeak {
    /// Frequency of the peak (Hz).
    pub freq_hz: f32,
    /// Power at the peak (arbitrary units).
    pub power: f32,
}

/// Detect spectral peaks in a time-series of LFP samples.
///
/// - `lfp_samples`: evenly-spaced LFP values (e.g. from `region_mean_lfp`).
/// - `dt_ms`: time step between samples (ms).
/// - `min_freq_hz`: ignore peaks below this frequency.
/// - `max_peaks`: maximum number of peaks to return (strongest first).
///
/// Returns peaks sorted by descending power.
pub fn detect_oscillation_peaks(
    lfp_samples: &[f32],
    dt_ms: f32,
    min_freq_hz: f32,
    max_peaks: usize,
) -> Vec<SpectralPeak> {
    let n = lfp_samples.len();
    if n < 4 {
        return Vec::new();
    }

    // Remove mean.
    let mean: f32 = lfp_samples.iter().sum::<f32>() / n as f32;
    let signal: Vec<f32> = lfp_samples.iter().map(|&v| v - mean).collect();

    // Apply Hann window to reduce spectral leakage.
    let windowed: Vec<f32> = signal
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
            v * w
        })
        .collect();

    // FFT via realfft.
    let mut planner = realfft::RealFftPlanner::new();
    let r2c = planner.plan_fft_forward(n);
    let mut input = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();
    input.copy_from_slice(&windowed);
    r2c.process(&mut input, &mut spectrum).expect("FFT failed");

    // Compute power spectrum (magnitude squared).
    let dt_s = dt_ms * 1e-3;
    let sample_rate = 1.0 / dt_s; // Hz
    let freq_resolution = sample_rate / n as f32;
    let n_bins = spectrum.len();

    let mut power_spectrum: Vec<f32> = Vec::with_capacity(n_bins);
    for bin in spectrum.iter() {
        let power = bin.re * bin.re + bin.im * bin.im;
        power_spectrum.push(power);
    }

    // Find local maxima above min_freq_hz.
    let min_bin = (min_freq_hz / freq_resolution).ceil() as usize;
    let mut peaks: Vec<SpectralPeak> = Vec::new();

    for i in min_bin..n_bins.saturating_sub(1) {
        if power_spectrum[i] > power_spectrum[i - 1]
            && power_spectrum[i] > power_spectrum[i + 1]
        {
            peaks.push(SpectralPeak {
                freq_hz: i as f32 * freq_resolution,
                power: power_spectrum[i],
            });
        }
    }

    // Sort by descending power, keep top max_peaks.
    peaks.sort_by(|a, b| b.power.partial_cmp(&a.power).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(max_peaks);
    peaks
}

// ── Functional connectivity matrix ──────────────────────────────────────

/// Compute a Pearson-correlation functional connectivity matrix from
/// binned spike-count time series.
///
/// - `spike_counts`: `[neuron][time_bin]` — spike counts per neuron per bin.
/// - Returns an `n × n` symmetric matrix of Pearson r-values in `[-1, 1]`.
///
/// Complexity: O(N² × T) where N = neurons, T = time bins.
pub fn functional_connectivity(spike_counts: &[Vec<u32>]) -> Vec<Vec<f32>> {
    let n = spike_counts.len();
    if n == 0 {
        return Vec::new();
    }

    // Pre-compute means and standard deviations.
    let means: Vec<f32> = spike_counts
        .iter()
        .map(|ts| {
            let sum: f32 = ts.iter().map(|&c| c as f32).sum();
            sum / ts.len().max(1) as f32
        })
        .collect();

    let stds: Vec<f32> = spike_counts
        .iter()
        .zip(means.iter())
        .map(|(ts, &mu)| {
            let var: f32 = ts.iter().map(|&c| (c as f32 - mu).powi(2)).sum::<f32>()
                / ts.len().max(1) as f32;
            var.sqrt().max(1e-10) // avoid division by zero
        })
        .collect();

    let mut matrix = vec![vec![0.0_f32; n]; n];
    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let t_len = spike_counts[i].len().min(spike_counts[j].len());
            if t_len == 0 {
                continue;
            }
            let mut cov = 0.0_f32;
            for t in 0..t_len {
                cov += (spike_counts[i][t] as f32 - means[i])
                    * (spike_counts[j][t] as f32 - means[j]);
            }
            cov /= t_len as f32;
            let r = cov / (stds[i] * stds[j]);
            let r_clamped = r.clamp(-1.0, 1.0);
            matrix[i][j] = r_clamped;
            matrix[j][i] = r_clamped;
        }
    }
    matrix
}
