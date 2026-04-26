//! Multi-layer perceptron convenience wrapper.

use crate::engine::value::Node;
use crate::nn::activations::Activations;
use crate::nn::layer::Layer;
use crate::nn::visualization::{render_network_to, NetworkVisualizationConfig};

/// Simple feed-forward neural network composed of sequential layers.
pub struct Mlp {
    layers: Vec<Layer>,
    layer_sizes: Vec<usize>,
}

impl Mlp {
    /// Construct an MLP from a list of layer widths.
    pub fn new(layer_widths: &[usize], activations: &[Activations]) -> Self {
        let mut mlp = Self {
            layers: Vec::new(),
            layer_sizes: layer_widths.to_vec(),
        };
        for i in 0..layer_widths.len() - 1 {
            let layer = Layer::new(layer_widths[i], layer_widths[i + 1], &activations[i]);
            mlp.layers.push(layer);
        }
        mlp
    }

    /// Evaluate the network on a single input example.
    pub fn forward(&self, inputs: &[Node]) -> Vec<Node> {
        let mut current = inputs.to_vec();
        for layer in self.layers.iter() {
            current = layer.forward(&current);
        }
        current
    }

    pub fn parameters(&self) -> Vec<Node> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
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
            names.push(format!("{}", layer.get_activation()));
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
