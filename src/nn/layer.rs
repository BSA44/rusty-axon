//! Compositions of neurons into layers.

use crate::engine::value::Value;
use crate::nn::neuron::Neuron;

/// A fully connected layer consisting of multiple neurons.
pub struct Layer {
    // TODO: store a collection of neurons and optional layer-level metadata.
}

impl Layer {
    /// Create a fully connected layer with the specified input/output sizes.
    pub fn new(_inputs: usize, _outputs: usize) -> Self {
        let _ = core::marker::PhantomData::<Neuron>;
        todo!("Instantiate `outputs` neurons each with `inputs` weights");
    }

    /// Compute the output activations for this layer.
    pub fn forward(&self, _inputs: &[Value]) -> Vec<Value> {
        todo!("Call forward on each neuron and collect the results");
    }
}
