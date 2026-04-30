//! Multi-layer perceptron convenience wrapper.
//!
//! Phase 3 swapped the internal `Vec<Layer>` (legacy scalar `Neuron` dot
//! products) for `Vec<Linear>` (fused [`MatMulTape`] per layer).  Phase 6
//! ungates the type so it's available in pure-inference builds: the
//! `Node`-based `forward` and parameter-list helpers stay behind
//! `cfg(feature = "train")`, while `save`, `load`, `infer`, and `infer_into`
//! are always-on.

use std::fs::File;
use std::io::{self, BufReader, BufWriter};
#[cfg(feature = "train")]
use std::ops::Range;
use std::path::Path;

#[cfg(feature = "train")]
use crate::engine::value::Node;
use crate::format::axn::{AxnReader, AxnWriter};
use crate::nn::activations::Activations;
use crate::nn::arena::InferArena;
use crate::nn::linear::Linear;
#[cfg(feature = "train")]
use crate::nn::visualization::{render_network_to, NetworkVisualizationConfig};

/// Simple feed-forward neural network composed of sequential `Linear` layers.
pub struct Mlp {
    layers: Vec<Linear>,
    layer_sizes: Vec<usize>,
}

impl Mlp {
    /// Construct an MLP from a list of layer widths.
    ///
    /// `layer_widths.len() == activations.len() + 1`: there is one `Linear`
    /// per gap between widths, each carrying the matching activation.
    pub fn new(layer_widths: &[usize], activations: &[Activations]) -> Self {
        assert!(
            layer_widths.len() >= 2,
            "Mlp requires at least an input and output width"
        );
        assert_eq!(
            activations.len(),
            layer_widths.len() - 1,
            "expected {} activations, got {}",
            layer_widths.len() - 1,
            activations.len()
        );

        let mut layers = Vec::with_capacity(layer_widths.len() - 1);
        for i in 0..layer_widths.len() - 1 {
            layers.push(Linear::new(
                layer_widths[i],
                layer_widths[i + 1],
                activations[i].clone(),
            ));
        }

        Self {
            layers,
            layer_sizes: layer_widths.to_vec(),
        }
    }

    /// Construct an MLP from caller-supplied `Linear` layers (test fixtures,
    /// `Mlp::load` in Phase 5, fine-tune helpers in Phase 11).  Validates
    /// that successive layers chain dimensionally.
    pub fn with_layers(layers: Vec<Linear>) -> Self {
        assert!(!layers.is_empty(), "Mlp::with_layers requires >=1 layer");
        for w in layers.windows(2) {
            assert_eq!(
                w[0].out_dim(),
                w[1].in_dim(),
                "layer dimensions do not chain: {} -> {} then {} -> {}",
                w[0].in_dim(),
                w[0].out_dim(),
                w[1].in_dim(),
                w[1].out_dim(),
            );
        }
        let mut layer_sizes = Vec::with_capacity(layers.len() + 1);
        layer_sizes.push(layers[0].in_dim());
        for l in &layers {
            layer_sizes.push(l.out_dim());
        }
        Self {
            layers,
            layer_sizes,
        }
    }

    /// Train-path forward.  Builds the `Node` graph that `loss.backward()`
    /// later walks.
    ///
    /// # Panics
    ///
    /// Panics if any layer is quantized (Phase 7's INT8 path is
    /// inference-only).  Use [`Mlp::infer`] / [`Mlp::infer_into`] to run a
    /// quantized model, or reload f32 weights to fine-tune.
    #[cfg(feature = "train")]
    pub fn forward(&self, inputs: &[Node]) -> Vec<Node> {
        assert!(
            !self.is_quantized(),
            "Cannot train a quantized Mlp; load f32 weights for fine-tuning"
        );
        let mut current = inputs.to_vec();
        for layer in self.layers.iter() {
            current = layer.forward(&current);
        }
        current
    }

    /// Whether **any** layer of this network is quantized (Phase 7).
    pub fn is_quantized(&self) -> bool {
        self.layers.iter().any(|l| l.is_quantized())
    }

    /// Quantize every layer to per-tensor symmetric INT8 in place (Phase 7).
    /// Frees the f32 weight buffers; biases stay f32.  After this call the
    /// model is inference-only — `forward` and `parameters` panic until
    /// f32 weights are reloaded.
    #[cfg(feature = "quant-i8")]
    pub fn quantize_to_i8(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.quantize_to_i8();
        }
    }

    /// Pure-`f32` inference: allocates the output `Vec<f32>` and runs the
    /// network forward.  Always available, including under `--features
    /// inference`.  Each call allocates two scratch buffers; use
    /// [`Mlp::infer_into`] (Phase 8) to avoid them in hot loops.
    pub fn infer(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(
            input.len(),
            self.layers[0].in_dim(),
            "input length must match the first layer's in_dim ({})",
            self.layers[0].in_dim()
        );
        let mut current: Vec<f32> = input.to_vec();
        let mut next: Vec<f32> = Vec::new();
        for layer in self.layers.iter() {
            next.clear();
            next.resize(layer.out_dim(), 0.0);
            layer.infer_into_f32(&current, &mut next);
            std::mem::swap(&mut current, &mut next);
        }
        current
    }

    /// Pure-`f32` inference writing into a caller-provided `output` slice.
    /// Allocates one internal scratch buffer of the largest hidden size.
    /// Phase 8 introduces a static-arena variant that drops this allocation
    /// entirely.
    ///
    /// # Panics
    ///
    /// Panics if `input.len()` or `output.len()` does not match the network's
    /// input / output dimensions.
    pub fn infer_into(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(
            input.len(),
            self.layers[0].in_dim(),
            "input length must match the first layer's in_dim ({})",
            self.layers[0].in_dim()
        );
        assert_eq!(
            output.len(),
            self.layers.last().unwrap().out_dim(),
            "output length must match the last layer's out_dim ({})",
            self.layers.last().unwrap().out_dim()
        );

        if self.layers.len() == 1 {
            self.layers[0].infer_into_f32(input, output);
            return;
        }

        // Two ping-pong scratch buffers sized to the largest hidden width.
        let mut max_hidden = 0;
        for l in &self.layers[..self.layers.len() - 1] {
            if l.out_dim() > max_hidden {
                max_hidden = l.out_dim();
            }
        }
        let mut buf_a = vec![0.0_f32; max_hidden];
        let mut buf_b = vec![0.0_f32; max_hidden];

        // First layer: input -> buf_a
        let first = &self.layers[0];
        let first_out = &mut buf_a[..first.out_dim()];
        first.infer_into_f32(input, first_out);

        // Middle layers ping-pong between buf_a / buf_b.
        let mut src_is_a = true;
        for layer in &self.layers[1..self.layers.len() - 1] {
            let in_dim = layer.in_dim();
            let out_dim = layer.out_dim();
            if src_is_a {
                let (src, dst) = (&buf_a[..in_dim], &mut buf_b[..out_dim]);
                layer.infer_into_f32(src, dst);
            } else {
                let (src, dst) = (&buf_b[..in_dim], &mut buf_a[..out_dim]);
                layer.infer_into_f32(src, dst);
            }
            src_is_a = !src_is_a;
        }

        // Final layer writes directly into `output`.
        let last = self.layers.last().unwrap();
        let last_in_dim = last.in_dim();
        if src_is_a {
            last.infer_into_f32(&buf_a[..last_in_dim], output);
        } else {
            last.infer_into_f32(&buf_b[..last_in_dim], output);
        }
    }

    /// Pure-`f32` inference using a caller-owned [`InferArena`] for every
    /// intermediate layer's scratch space.  Allocates **zero** bytes per call
    /// once the arena is built — the headline edge-inference path for the
    /// paper's latency table.
    ///
    /// The arena must have been built with [`InferArena::for_mlp`] for an
    /// `Mlp` of identical shape.  Mismatched shapes panic.
    ///
    /// # Panics
    ///
    /// Panics if `input.len()` or `output.len()` does not match the network's
    /// input / output dimensions, or if `arena.slots.len() != num_layers - 1`.
    pub fn infer_into_arena(
        &self,
        input: &[f32],
        output: &mut [f32],
        arena: &mut InferArena,
    ) {
        let n = self.layers.len();
        assert_eq!(
            input.len(),
            self.layers[0].in_dim(),
            "input length must match the first layer's in_dim ({})",
            self.layers[0].in_dim()
        );
        assert_eq!(
            output.len(),
            self.layers[n - 1].out_dim(),
            "output length must match the last layer's out_dim ({})",
            self.layers[n - 1].out_dim()
        );
        assert_eq!(
            arena.slots.len(),
            n.saturating_sub(1),
            "arena slot count {} does not match {} intermediate layers; \
             rebuild the arena with InferArena::for_mlp(&mlp)",
            arena.slots.len(),
            n.saturating_sub(1)
        );

        // Single-layer fast path: no intermediates needed.
        if n == 1 {
            self.layers[0].infer_into_f32(input, output);
            return;
        }

        // Layer 0: input -> arena.buffer[slots[0]].
        {
            let slot0 = arena.slots[0].clone();
            let out_slice = &mut arena.buffer[slot0];
            self.layers[0].infer_into_f32(input, out_slice);
        }

        // Middle layers: arena.buffer[slots[i-1]] -> arena.buffer[slots[i]].
        // Slots are allocated contiguously and in order, so slots[i].start ==
        // slots[i-1].end.  `split_at_mut` at slots[i].start lets us hold the
        // previous slot immutably and the current one mutably without unsafe.
        for i in 1..n - 1 {
            let in_range = arena.slots[i - 1].clone();
            let out_range = arena.slots[i].clone();
            debug_assert_eq!(
                in_range.end, out_range.start,
                "InferArena slots are expected to be contiguous"
            );
            let split = out_range.start;
            let out_len = out_range.end - out_range.start;
            let (head, tail) = arena.buffer.split_at_mut(split);
            let in_slice = &head[in_range];
            let out_slice = &mut tail[..out_len];
            self.layers[i].infer_into_f32(in_slice, out_slice);
        }

        // Last layer: arena.buffer[slots[n-2]] -> output.
        let last_in_range = arena.slots[n - 2].clone();
        let last_in_slice = &arena.buffer[last_in_range];
        self.layers[n - 1].infer_into_f32(last_in_slice, output);
    }

    /// All trainable parameters across every layer.  Order: layer-0 weights,
    /// layer-0 biases, layer-1 weights, layer-1 biases, ...
    #[cfg(feature = "train")]
    pub fn parameters(&self) -> Vec<Node> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    /// Borrow one layer of the network.
    ///
    /// # Panics
    /// Panics if `idx >= num_layers() - 1` (i.e. out of the layer range, not
    /// the width range).
    pub fn layer(&self, idx: usize) -> &Linear {
        &self.layers[idx]
    }

    /// Parameters from a contiguous slice of layers, e.g.
    /// `mlp.parameters_for_layers(2..3)` for last-layer-only fine-tune
    /// (Phase 11 demo target).
    ///
    /// # Panics
    /// Panics if `range` falls outside `0..num_linear_layers()`.
    #[cfg(feature = "train")]
    pub fn parameters_for_layers(&self, range: Range<usize>) -> Vec<Node> {
        assert!(
            range.end <= self.layers.len(),
            "parameters_for_layers: range {:?} exceeds {} layers",
            range,
            self.layers.len()
        );
        self.layers[range]
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    /// Number of `Linear` layers (i.e. `layer_widths.len() - 1`).
    pub fn num_linear_layers(&self) -> usize {
        self.layers.len()
    }

    /// Serialize the network to an `.axn` file.
    ///
    /// Writes one `layer{N}.weight` (row-major `[out_dim, in_dim]`, F32) and
    /// one `layer{N}.bias` (`[out_dim]`, F32) per `Linear`.  Activation
    /// choices are **not** stored in v1; callers pass them back in to
    /// [`Mlp::load`].
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let mut axn = AxnWriter::new(writer);
        for (i, layer) in self.layers.iter().enumerate() {
            let weight_name = format!("layer{}.weight", i);
            let bias_name = format!("layer{}.bias", i);
            let dims = [layer.out_dim() as u32, layer.in_dim() as u32];
            #[cfg(feature = "quant-i8")]
            if let Some((qweights, scale)) = layer.quantized_weights() {
                axn.add_tensor_i8(&weight_name, &dims, scale, qweights);
                axn.add_tensor_f32(&bias_name, &[layer.out_dim() as u32], &layer.bias());
                continue;
            }
            axn.add_tensor_f32(&weight_name, &dims, &layer.weights());
            axn.add_tensor_f32(&bias_name, &[layer.out_dim() as u32], &layer.bias());
        }
        axn.finish()?;
        Ok(())
    }

    /// Quantize-then-save shortcut: equivalent to `quantize_to_i8()` followed
    /// by `save()`.  Mutates `self`.
    #[cfg(feature = "quant-i8")]
    pub fn save_quantized(&mut self, path: &Path) -> io::Result<()> {
        self.quantize_to_i8();
        self.save(path)
    }

    /// Reconstruct an `Mlp` from an `.axn` file.  `activations.len()` must
    /// match the number of `Linear` layers found on disk.
    pub fn load(path: &Path, activations: &[Activations]) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = AxnReader::open(BufReader::new(file))?;

        // Pair tensors by layer index using the `layer{N}.{weight|bias}` convention.
        let metas: Vec<_> = reader.tensors().to_vec();
        let num_layers = metas.len() / 2;
        if metas.len() != num_layers * 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected 2 tensors per layer (weight + bias)",
            ));
        }
        if activations.len() != num_layers {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "activation count {} does not match {} layers in `{}`",
                    activations.len(),
                    num_layers,
                    path.display()
                ),
            ));
        }

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let w_name = format!("layer{}.weight", i);
            let b_name = format!("layer{}.bias", i);
            let w_idx = metas
                .iter()
                .position(|m| m.name == w_name)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("missing tensor `{}`", w_name),
                    )
                })?;
            let b_idx = metas
                .iter()
                .position(|m| m.name == b_name)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("missing tensor `{}`", b_name),
                    )
                })?;
            let w_meta = &metas[w_idx];
            let b_meta = &metas[b_idx];
            if w_meta.dims.len() != 2 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("`{}` must be rank-2", w_name),
                ));
            }
            if b_meta.dims.len() != 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("`{}` must be rank-1", b_name),
                ));
            }
            let out_dim = w_meta.dims[0] as usize;
            let in_dim = w_meta.dims[1] as usize;
            if b_meta.dims[0] as usize != out_dim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "`{}` out_dim {} does not match `{}` length {}",
                        w_name, out_dim, b_name, b_meta.dims[0]
                    ),
                ));
            }
            let bias = reader.read_tensor_f32(b_idx)?;
            // Phase 7: weight tensors may be either F32 or I8.  Dispatch on
            // the dtype we parsed from the header.
            match w_meta.dtype {
                crate::format::axn::Dtype::F32 => {
                    let weights = reader.read_tensor_f32(w_idx)?;
                    layers.push(Linear::with_weights(
                        in_dim,
                        out_dim,
                        weights,
                        bias,
                        activations[i].clone(),
                    ));
                }
                crate::format::axn::Dtype::I8 => {
                    #[cfg(feature = "quant-i8")]
                    {
                        let (qweights, scale) = reader.read_tensor_i8(w_idx)?;
                        layers.push(Linear::with_quantized_weights(
                            in_dim,
                            out_dim,
                            qweights,
                            scale,
                            bias,
                            activations[i].clone(),
                        ));
                    }
                    #[cfg(not(feature = "quant-i8"))]
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "`{}` is INT8-quantized but the `quant-i8` feature \
                                 is disabled in this build",
                                w_name
                            ),
                        ));
                    }
                }
            }
        }

        Ok(Self::with_layers(layers))
    }

    /// Generate layer names for visualization
    #[cfg(feature = "train")]
    fn generate_layer_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        // Input layer
        names.push("Input Layer".to_string());

        // Hidden layers
        for i in 1..self.layer_sizes.len() - 1 {
            names.push(format!("Hidden Layer {}", i));
        }

        // Output layer
        if self.layer_sizes.len() > 1 {
            names.push("Output Layer".to_string());
        }

        names
    }

    /// Generate activation function names for visualization
    #[cfg(feature = "train")]
    fn generate_activation_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        // Input layer has no activation
        names.push(String::new());

        // Each subsequent layer has an activation from the corresponding layer
        for layer in &self.layers {
            names.push(format!("{}", layer.activation()));
        }

        names
    }

    /// Visualize the network architecture as a layer-oriented graph
    ///
    /// # Example
    /// ```ignore
    /// let mlp = Mlp::new(&[2, 4, 4, 1], &[Activations::Tanh, Activations::Tanh, Activations::Sigmoid]);
    /// mlp.visualize_network("my_network", "png").unwrap();
    /// ```
    #[cfg(feature = "train")]
    pub fn visualize_network(&self, output_name: &str, format: &str) -> std::io::Result<()> {
        let config = NetworkVisualizationConfig::default();
        self.visualize_network_with_config(output_name, format, &config)
    }

    /// Visualize the network architecture with custom configuration
    ///
    /// # Example
    /// ```ignore
    /// use rusty_axon::nn::visualization::NetworkVisualizationConfig;
    ///
    /// let config = NetworkVisualizationConfig::with_colors(
    ///     "lavender", "mediumpurple",  // Input layer
    ///     "mistyrose", "lightcoral",   // Hidden layers
    ///     "lightcyan", "lightskyblue", // Output layer
    /// );
    ///
    /// mlp.visualize_network_with_config("my_network", "png", &config).unwrap();
    /// ```
    #[cfg(feature = "train")]
    pub fn visualize_network_with_config(
        &self,
        output_name: &str,
        format: &str,
        config: &NetworkVisualizationConfig,
    ) -> std::io::Result<()> {
        let layer_names = self.generate_layer_names();
        let activation_names = self.generate_activation_names();
        render_network_to(
            output_name,
            format,
            &self.layer_sizes,
            &layer_names,
            &activation_names,
            config,
        )
    }

    /// Render network architecture to PNG (convenience method)
    #[cfg(feature = "train")]
    pub fn render_network_png(&self, output_name: &str) -> std::io::Result<()> {
        self.visualize_network(output_name, "png")
    }

    /// Render network architecture to SVG (convenience method)
    #[cfg(feature = "train")]
    pub fn render_network_svg(&self, output_name: &str) -> std::io::Result<()> {
        self.visualize_network(output_name, "svg")
    }

    /// Render network architecture to PDF (convenience method)
    #[cfg(feature = "train")]
    pub fn render_network_pdf(&self, output_name: &str) -> std::io::Result<()> {
        self.visualize_network(output_name, "pdf")
    }

    /// Get layer information
    pub fn get_architecture(&self) -> &[usize] {
        &self.layer_sizes
    }

    /// Get number of layers (including input layer)
    pub fn num_layers(&self) -> usize {
        self.layer_sizes.len()
    }
}
