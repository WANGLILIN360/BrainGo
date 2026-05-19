//! Sparse quantisation & compression for large connectomes.
//!
//! Design doc §16.4: synapse weight quantisation (8-bit or 16-bit) and
//! run-length encoding of CSR index arrays. Intended for the fruit-fly
//! scale (140k neurons) where the full `.braindb` file would otherwise
//! exceed several GB.
//!
//! This is a **stub module** — full implementation is deferred to Phase 2
//! (fruit fly). The API surface below is provided so that downstream code
//! can reference it without conditional compilation.

/// Placeholder for a compressed CSR segment.
#[derive(Clone, Debug, Default)]
pub struct CompressedCSR {
    /// Number of neurons in the compressed segment.
    pub n_neurons: u32,
    /// Compressed row-pointer bytes (RLE-encoded).
    pub row_ptr_bytes: Vec<u8>,
    /// Compressed column-index bytes (varint-encoded).
    pub col_idx_bytes: Vec<u8>,
    /// Quantised weight bytes (1 or 2 bytes per synapse).
    pub weight_bytes: Vec<u8>,
    /// Bits per weight (8 or 16).
    pub weight_bits: u8,
}

impl CompressedCSR {
    /// Create an empty compressed CSR.
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimated decompressed size in bytes.
    pub fn decompressed_size(&self) -> usize {
        self.row_ptr_bytes.len() + self.col_idx_bytes.len() + self.weight_bytes.len()
    }

    /// Compression ratio (compressed / estimated-original).
    pub fn compression_ratio(&self) -> f32 {
        let original = self.decompressed_size() as f32;
        if original == 0.0 { return 1.0; }
        let compressed = (self.row_ptr_bytes.len()
            + self.col_idx_bytes.len()
            + self.weight_bytes.len()) as f32;
        compressed / original
    }
}

/// Quantise a slice of f32 weights into 8-bit values.
/// Maps [0, max_weight] → [0, 255].
pub fn quantise_weights_8bit(weights: &[f32], max_weight: f32) -> Vec<u8> {
    let scale = if max_weight > 0.0 { 255.0 / max_weight } else { 0.0 };
    weights.iter().map(|&w| (w * scale).min(255.0) as u8).collect()
}

/// Dequantise 8-bit weights back to f32.
pub fn dequantise_weights_8bit(bytes: &[u8], max_weight: f32) -> Vec<f32> {
    let scale = max_weight / 255.0;
    bytes.iter().map(|&b| b as f32 * scale).collect()
}
