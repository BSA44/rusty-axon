//! Multi-layer perceptron convenience wrapper.

use crate::engine::value::Value;
use crate::nn::layer::Layer;

/// Simple feed-forward neural network composed of sequential layers.
pub struct Mlp {
    // TODO: maintain ordered layers and optional output activation.
}

impl Mlp {
    /// Construct an MLP from a list of layer widths.
    pub fn new(_widths: &[usize]) -> Self {
        let _ = core::marker::PhantomData::<Layer>;
        todo!("Create layers connecting each successive pair of widths");
    }

    /// Evaluate the network on a single input example.
    pub fn forward(&self, _inputs: &[Value]) -> Vec<Value> {
        todo!("Sequentially apply each layer to the inputs");
    }
}
