//! Multi-layer perceptron convenience wrapper.
//!
//! Phase 3 swaps the internal `Vec<Layer>` (legacy scalar `Neuron` dot
//! products) for `Vec<Linear>` (fused [`MatMulTape`] per layer).  The public
//! API — `Mlp::new`, `forward`, `parameters`, the visualization helpers —
//! is unchanged so every existing example continues to build and run.  The
//! legacy `Layer`/`Neuron` modules are kept on disk as the scalar baseline
//! that Phase 8 benchmarks against.

use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::ops::Range;
use std::path::Path;

use crate::engine::value::Node;
use crate::format::axn::{AxnReader, AxnWriter};
use crate::nn::activations::Activations;
use crate::nn::linear::Linear;
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

    /// Evaluate the network on a single input example.
    pub fn forward(&self, inputs: &[Node]) -> Vec<Node> {
        let mut current = inputs.to_vec();
        for layer in self.layers.iter() {
            current = layer.forward(&current);
        }
        current
    }

    /// All trainable parameters across every layer.  Order: layer-0 weights,
    /// layer-0 biases, layer-1 weights, layer-1 biases, ...
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
            axn.add_tensor_f32(&weight_name, &dims, &layer.weights());
            axn.add_tensor_f32(&bias_name, &[layer.out_dim() as u32], &layer.bias());
        }
        axn.finish()?;
        Ok(())
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
            let weights = reader.read_tensor_f32(w_idx)?;
            let bias = reader.read_tensor_f32(b_idx)?;
            layers.push(Linear::with_weights(
                in_dim,
                out_dim,
                weights,
                bias,
                activations[i].clone(),
            ));
        }

        Ok(Self::with_layers(layers))
    }

    /// Generate layer names for visualization
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
    pub fn render_network_png(&self, output_name: &str) -> std::io::Result<()> {
        self.visualize_network(output_name, "png")
    }

    /// Render network architecture to SVG (convenience method)
    pub fn render_network_svg(&self, output_name: &str) -> std::io::Result<()> {
        self.visualize_network(output_name, "svg")
    }

    /// Render network architecture to PDF (convenience method)
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
