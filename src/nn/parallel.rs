//! Parallel batch training utilities using data parallelism.
//!
//! This module provides parallel training capabilities by processing multiple
//! training examples simultaneously across threads. Each thread creates its own
//! local computation graph, computes gradients independently, and then gradients
//! are averaged and applied to the master network.
//!
//! # Example
//! ```ignore
//! use rusty_axon::nn::parallel::ParallelTrainer;
//! use rusty_axon::nn::mlp::Mlp;
//! use rusty_axon::nn::activations::Activations;
//! use rusty_axon::loss::mse::MeanSquaredError;
//!
//! let mut mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
//! let trainer = ParallelTrainer::new(0.5, vec![2, 4, 1], vec![Activations::Tanh, Activations::Sigmoid]);
//! let loss_fn = MeanSquaredError;
//!
//! let batch = vec![
//!     (vec![0.0, 0.0], vec![0.0]),
//!     (vec![0.0, 1.0], vec![1.0]),
//! ];
//!
//! let loss = trainer.train_batch(&mut mlp, &batch, &loss_fn);
//! ```

use rayon::prelude::*;
use crate::engine::Node;
use crate::nn::mlp::Mlp;
use crate::nn::activations::Activations;
use crate::loss::loss::Loss;

/// Parallel trainer for batch processing using data parallelism.
///
/// Each training example in a batch is processed in parallel by a separate thread.
/// Gradients are accumulated and averaged, then applied to update the master network.
pub struct ParallelTrainer {
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Network architecture (layer sizes)
    architecture: Vec<usize>,
    /// Activation functions for each layer
    activations: Vec<Activations>,
}

impl ParallelTrainer {
    /// Create a new parallel trainer.
    ///
    /// # Arguments
    /// * `learning_rate` - Step size for gradient descent
    /// * `architecture` - Layer sizes (e.g., `vec![2, 4, 1]` for 2 inputs, 4 hidden, 1 output)
    /// * `activations` - Activation functions for each layer transition
    pub fn new(
        learning_rate: f64,
        architecture: Vec<usize>,
        activations: Vec<Activations>,
    ) -> Self {
        Self {
            learning_rate,
            architecture,
            activations,
        }
    }

    /// Create a parallel trainer from an existing MLP.
    pub fn from_mlp(learning_rate: f64, mlp: &Mlp) -> Self {
        Self {
            learning_rate,
            architecture: mlp.get_architecture().to_vec(),
            activations: mlp.get_activations(),
        }
    }

    /// Train on a batch of examples in parallel.
    ///
    /// This method:
    /// 1. Extracts current weights from the master MLP
    /// 2. Spawns parallel workers, each with a local MLP copy
    /// 3. Each worker computes forward/backward pass for its example(s)
    /// 4. Collects and averages gradients from all workers
    /// 5. Updates master MLP weights
    ///
    /// # Arguments
    /// * `mlp` - The master MLP to train
    /// * `batch` - Batch of (inputs, targets) pairs
    /// * `loss_fn` - Loss function to use
    ///
    /// # Returns
    /// Average loss over the batch
    pub fn train_batch<L: Loss + Sync>(
        &self,
        mlp: &mut Mlp,
        batch: &[(Vec<f64>, Vec<f64>)],
        loss_fn: &L,
    ) -> f64 {
        if batch.is_empty() {
            return 0.0;
        }

        // 1. Extract current weights from master network
        let current_weights = mlp.get_weights();
        let arch = &self.architecture;
        let acts = &self.activations;

        // 2-3. Parallel forward/backward for each example
        let results: Vec<(Vec<f64>, f64)> = batch
            .par_iter()
            .map(|(inputs, targets)| {
                // Create local MLP with current weights (each thread gets its own copy)
                let local_mlp = Mlp::with_weights(arch, acts, &current_weights);

                // Convert inputs and targets to Nodes
                let input_nodes: Vec<Node> = inputs.iter()
                    .map(|&x| Node::from(x))
                    .collect();
                let target_nodes: Vec<Node> = targets.iter()
                    .map(|&t| Node::from(t))
                    .collect();

                // Forward pass
                let outputs = local_mlp.forward(&input_nodes);

                // Compute loss
                let mut loss = loss_fn.forward(&outputs, &target_nodes);
                let loss_val = loss.get_value();

                // Backward pass - compute gradients
                loss.backward();

                // Collect gradients from local parameters
                let gradients: Vec<f64> = local_mlp.parameters()
                    .iter()
                    .map(|p| p.get_gradient())
                    .collect();

                (gradients, loss_val)
            })
            .collect();

        // 4. Sum gradients across all examples (matches sequential SGD behavior)
        let num_examples = results.len() as f64;
        let num_params = results[0].0.len();

        // SUM gradients (not average) to match sequential batch training behavior
        let sum_gradients: Vec<f64> = (0..num_params)
            .map(|i| {
                results.iter().map(|(g, _)| g[i]).sum::<f64>()
            })
            .collect();

        let avg_loss: f64 = results.iter().map(|(_, l)| l).sum::<f64>() / num_examples;

        // 5. Update master MLP weights using summed gradients
        let current_weights = mlp.get_weights();
        let new_weights: Vec<f64> = current_weights
            .iter()
            .zip(sum_gradients.iter())
            .map(|(w, g)| w - self.learning_rate * g)
            .collect();
        mlp.set_weights(&new_weights);

        avg_loss
    }

    /// Train on a batch with a simple MSE loss (convenience method).
    ///
    /// For single-output regression problems.
    pub fn train_batch_mse(
        &self,
        mlp: &mut Mlp,
        batch: &[(Vec<f64>, f64)],
    ) -> f64 {
        if batch.is_empty() {
            return 0.0;
        }

        let current_weights = mlp.get_weights();
        let arch = &self.architecture;
        let acts = &self.activations;

        let results: Vec<(Vec<f64>, f64)> = batch
            .par_iter()
            .map(|(inputs, target)| {
                let local_mlp = Mlp::with_weights(arch, acts, &current_weights);

                let input_nodes: Vec<Node> = inputs.iter()
                    .map(|&x| Node::from(x))
                    .collect();

                let outputs = local_mlp.forward(&input_nodes);
                let prediction = outputs[0].clone();

                // MSE loss for single output
                let diff = prediction - Node::from(*target);
                let mut loss = diff.pow(2.0);
                let loss_val = loss.get_value();

                loss.backward();

                let gradients: Vec<f64> = local_mlp.parameters()
                    .iter()
                    .map(|p| p.get_gradient())
                    .collect();

                (gradients, loss_val)
            })
            .collect();

        let num_examples = results.len() as f64;
        let num_params = results[0].0.len();

        // SUM gradients (not average) to match sequential batch training behavior
        let sum_gradients: Vec<f64> = (0..num_params)
            .map(|i| {
                results.iter().map(|(g, _)| g[i]).sum::<f64>()
            })
            .collect();

        let avg_loss: f64 = results.iter().map(|(_, l)| l).sum::<f64>() / num_examples;

        let current_weights = mlp.get_weights();
        let new_weights: Vec<f64> = current_weights
            .iter()
            .zip(sum_gradients.iter())
            .map(|(w, g)| w - self.learning_rate * g)
            .collect();
        mlp.set_weights(&new_weights);

        avg_loss
    }

    /// Set the learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    /// Get the current learning rate
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

/// Configure the number of threads for parallel training.
///
/// By default, Rayon uses all available CPU cores.
///
/// # Example
/// ```ignore
/// use rusty_axon::nn::parallel::set_num_threads;
/// set_num_threads(4); // Use 4 threads
/// ```
pub fn set_num_threads(num_threads: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .ok(); // Ignore error if already initialized
}

/// Get the number of threads Rayon will use.
pub fn get_num_threads() -> usize {
    rayon::current_num_threads()
}
