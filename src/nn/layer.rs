//! Compositions of neurons into layers.

use crate::engine::value::Node;
use crate::nn::neuron::Neuron;
use crate::nn::activations::Activations;
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
        self.neurons.iter()
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

    /// Get all weights as a flat Vec<f64> (all neurons concatenated)
    pub fn get_weights(&self) -> Vec<f64> {
        self.neurons.iter()
            .flat_map(|n| n.get_weights())
            .collect()
    }

    /// Create a layer with specific weight values
    pub fn with_weights(
        input_size: usize,
        weights: &[f64],
        activation: &Activations
    ) -> Self {
        let weights_per_neuron = input_size + 1; // +1 for bias
        let neurons = weights.chunks(weights_per_neuron)
            .map(|chunk| Neuron::with_weights(chunk, activation))
            .collect();
        
        Self { neurons, activation: activation.clone() }
    }

    /// Set weights from a flat Vec<f64>
    pub fn set_weights(&mut self, weights: &[f64]) {
        let weights_per_neuron = if self.neurons.is_empty() {
            return;
        } else {
            self.neurons[0].num_inputs() + 1
        };
        
        for (neuron, chunk) in self.neurons.iter_mut().zip(weights.chunks(weights_per_neuron)) {
            neuron.set_weights(chunk);
        }
    }
}
