//! Single neuron abstraction composed of weighted inputs and an activation.
use rand::Rng;
use crate::engine::value::Node;
use crate::nn::activations::Activations;
/// Basic neuron that consumes a vector of inputs and produces a scalar output.
pub struct Neuron {
    weights: Vec<Node>,
    bias: Node,
    activation: Activations,
}

impl Neuron {
    /// Create a new neuron with the requested number of inputs.
    pub fn new( num_inputs: usize, activation: Activations) -> Self{
        let mut rng = rand::rng();
        let mut neuron = Self {
            weights: Vec::new(),
            bias:Node::new(rng.random_range(-1.0..1.0)),
            activation: activation,
        };
        for _ in 0..num_inputs {
            let weight = Node::new(rng.random_range(-1.0..1.0));
            neuron.weights.push(weight);
        }

        neuron
    }

    /// Execute the forward pass for this neuron.
    pub fn forward(&self, inputs: &[Node]) -> Node {
        let mut weighted_sum = Node::new(0.0);
        for (input, weight) in inputs.iter().zip(self.weights.iter()) {
            weighted_sum = weighted_sum + input.clone() * weight.clone();
        }
        weighted_sum = weighted_sum + self.bias.clone();
        //todo add activation function
        self.activation.apply(weighted_sum)
    }

    pub fn parameters(&self) -> Vec<Node> {
        self.weights.iter()
        .cloned()
        .chain(std::iter::once(self.bias.clone()))
        .collect()
    }

    /// Get all weights and bias as plain f64 values (weights first, then bias)
    pub fn get_weights(&self) -> Vec<f64> {
        let mut weights: Vec<f64> = self.weights.iter()
            .map(|w| w.get_value())
            .collect();
        weights.push(self.bias.get_value());
        weights
    }

    /// Create a neuron with specific weight values (last value is bias)
    pub fn with_weights(weights: &[f64], activation: &Activations) -> Self {
        let n = weights.len() - 1; // last is bias
        Self {
            weights: weights[..n].iter().map(|&w| Node::from(w)).collect(),
            bias: Node::from(weights[n]),
            activation: activation.clone(),
        }
    }

    /// Set weights from f64 slice (last value is bias)
    pub fn set_weights(&mut self, weights: &[f64]) {
        for (w, &val) in self.weights.iter_mut().zip(weights.iter()) {
            w.set_value(val);
        }
        if weights.len() > self.weights.len() {
            self.bias.set_value(weights[self.weights.len()]);
        }
    }

    /// Get the number of input connections
    pub fn num_inputs(&self) -> usize {
        self.weights.len()
    }
}
