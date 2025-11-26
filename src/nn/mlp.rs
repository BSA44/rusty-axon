//! Multi-layer perceptron convenience wrapper.

use crate::engine::value::Node;
use crate::nn::activations::Activations;
use crate::nn::layer::Layer;

/// Simple feed-forward neural network composed of sequential layers.
pub struct Mlp {
    layers: Vec<Layer>,
}

impl Mlp {
    /// Construct an MLP from a list of layer widths.
    pub fn new(layer_widths: &[usize], activations: &[Activations]) -> Self {
        let mut mlp = Self {
            layers: Vec::new(),
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
        self.layers.iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
