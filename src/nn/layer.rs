//! Compositions of neurons into layers.

use crate::engine::value::Node;
use crate::nn::activations::Activations;
use crate::nn::neuron::Neuron;
/// A fully connected layer consisting of multiple neurons.
pub struct Layer {
    neurons: Vec<Neuron>,
    activation: Activations,
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
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }

    /// Get the activation function used by this layer
    pub fn get_activation(&self) -> &Activations {
        &self.activation
    }

    /// Get the number of neurons (output size) in this layer
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }
}
