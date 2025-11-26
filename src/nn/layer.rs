//! Compositions of neurons into layers.

use crate::engine::value::Node;
use crate::nn::neuron::Neuron;
use crate::nn::activations::Activations;
/// A fully connected layer consisting of multiple neurons.
pub struct Layer {
    neurons: Vec<Neuron>,
    activation: Activations,
    // TODO: store a collection of neurons and optional layer-level metadata.
}

impl Layer {
    /// Create a fully connected layer with the specified input/output sizes.
    pub fn new(num_inputs: usize, num_outputs: usize, activation: &Activations) -> Self {
        let mut layer = Self {
            neurons: Vec::new(),
            activation: activation.clone(),
        };
        for _ in 0..num_outputs {
            let neuron = Neuron::new(num_inputs, activation.clone());
            layer.neurons.push(neuron);
        }
        layer
    }

    /// Compute the output activations for this layer.
    pub fn forward(&self, inputs: &[Node]) -> Vec<Node> {
        let mut outputs = Vec::new();
        for neuron in self.neurons.iter() {
            outputs.push(neuron.forward(inputs));
        }
        outputs
    }

    pub fn parameters(&self) -> Vec<Node> {
        self.neurons.iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }
}
