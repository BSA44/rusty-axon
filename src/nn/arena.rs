//! Static inference arena (Phase 8).
//!
//! [`InferArena`] holds the per-layer scratch buffers for [`Mlp::infer_into_arena`]
//! ahead of time, so a hot inference loop allocates exactly zero bytes per
//! call.  The arena is sized once via [`InferArena::for_mlp`] and then reused
//! across requests; intermediates for layer `i` live in `arena.buffer[arena.slots[i]]`.
//!
//! Layout: one slot per layer output **except the last** — the final layer
//! writes directly into the caller's `output` buffer.  For an `N`-layer
//! network there are `N - 1` intermediate slots, packed contiguously in a
//! single `Vec<f32>`.  A 1-layer network has zero slots.
//!
//! [`Mlp::infer_into_arena`]: crate::nn::mlp::Mlp::infer_into_arena

use std::ops::Range;

use crate::nn::mlp::Mlp;

/// Pre-allocated scratch space for [`Mlp::infer_into_arena`].  Zero heap
/// allocation per inference call once constructed.
///
/// Always-on (available under both `train` and `inference` builds).
///
/// [`Mlp::infer_into_arena`]: crate::nn::mlp::Mlp::infer_into_arena
pub struct InferArena {
    /// Backing buffer: contiguous storage for every intermediate layer output.
    pub(crate) buffer: Vec<f32>,
    /// One range per intermediate layer output (i.e. layers `0..num_layers-1`).
    /// Empty for a 1-layer network.
    pub(crate) slots: Vec<Range<usize>>,
}

impl InferArena {
    /// Build an arena sized for `mlp`'s architecture.  Slot `i` covers
    /// `mlp.layer(i).out_dim()` floats; total buffer is the sum of the first
    /// `N - 1` layer widths.
    pub fn for_mlp(mlp: &Mlp) -> Self {
        let n = mlp.num_linear_layers();
        let num_intermediate = n.saturating_sub(1);
        let mut slots = Vec::with_capacity(num_intermediate);
        let mut total = 0_usize;
        for i in 0..num_intermediate {
            let size = mlp.layer(i).out_dim();
            slots.push(total..total + size);
            total += size;
        }
        Self {
            buffer: vec![0.0_f32; total],
            slots,
        }
    }

    /// Total bytes the arena owns (`buffer.len() * 4`).  Useful for the
    /// memory-footprint table in the paper.
    pub fn buffer_bytes(&self) -> usize {
        self.buffer.len() * std::mem::size_of::<f32>()
    }

    /// Number of intermediate slots (`= num_linear_layers - 1`).
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::activations::Activations;

    #[test]
    fn for_mlp_sizes_slots_correctly() {
        let mlp = Mlp::new(
            &[8, 16, 4, 2],
            &[Activations::ReLU, Activations::ReLU, Activations::None],
        );
        let arena = InferArena::for_mlp(&mlp);
        // 3 layers -> 2 intermediate slots: layer-0 out (16), layer-1 out (4).
        assert_eq!(arena.slots.len(), 2);
        assert_eq!(arena.slots[0], 0..16);
        assert_eq!(arena.slots[1], 16..20);
        assert_eq!(arena.buffer.len(), 20);
        assert_eq!(arena.buffer_bytes(), 20 * 4);
    }

    #[test]
    fn for_mlp_single_layer_has_no_slots() {
        let mlp = Mlp::new(&[4, 2], &[Activations::None]);
        let arena = InferArena::for_mlp(&mlp);
        assert_eq!(arena.slots.len(), 0);
        assert_eq!(arena.buffer.len(), 0);
    }
}
