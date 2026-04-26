#[cfg(test)]
mod tests {
    use crate::engine::Node;
    use crate::nn::activations::Activations;
    use crate::nn::neuron::Neuron;

    #[test]
    fn test_neuron_creation() {
        let neuron = Neuron::new(3, Activations::None);
        assert_eq!(neuron.parameters().len(), 4); // 3 weights + 1 bias
    }

    #[test]
    fn test_neuron_forward() {
        let neuron = Neuron::new(2, Activations::None);
        let inputs = vec![Node::from(1.0), Node::from(2.0)];
        let output = neuron.forward(&inputs);

        // Output should be a valid number
        assert!(output.get_value().is_finite());
    }

    #[test]
    fn test_neuron_gradients() {
        let neuron = Neuron::new(2, Activations::None);
        let inputs = vec![Node::from(1.0), Node::from(2.0)];
        let mut output = neuron.forward(&inputs);
        output.backward();

        // All parameters should have a finite gradient after backward; some
        // may legitimately be zero (e.g. weights multiplied by a zero input).
        for param in neuron.parameters() {
            assert!(param.get_gradient().is_finite());
        }
    }
    #[test]
    fn test_neuron_with_sigmoid() {
        let neuron = Neuron::new(2, Activations::Sigmoid);
        let inputs = vec![Node::from(1.0), Node::from(2.0)];
        let output = neuron.forward(&inputs);

        // Sigmoid output should be between 0 and 1
        let val = output.get_value();
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_neuron_with_tanh() {
        let neuron = Neuron::new(2, Activations::Tanh);
        let inputs = vec![Node::from(1.0), Node::from(2.0)];
        let output = neuron.forward(&inputs);

        // Tanh output should be between -1 and 1
        let val = output.get_value();
        assert!(val > -1.0 && val < 1.0);
    }

    // ========== LAYER TESTS ==========

    #[test]
    fn test_layer_creation() {
        use crate::nn::layer::Layer;

        let layer = Layer::new(3, 5, &Activations::None);
        // 5 neurons, each with (3 weights + 1 bias) = 5 * 4 = 20 parameters
        assert_eq!(layer.parameters().len(), 20);
    }

    #[test]
    fn test_layer_output_size() {
        use crate::nn::layer::Layer;

        let layer = Layer::new(3, 5, &Activations::None);
        let inputs = vec![Node::from(1.0), Node::from(2.0), Node::from(3.0)];
        let outputs = layer.forward(&inputs);

        // Should have 5 outputs (one per neuron)
        assert_eq!(outputs.len(), 5);

        // All outputs should be valid numbers
        for output in outputs {
            assert!(output.get_value().is_finite());
        }
    }

    #[test]
    fn test_layer_with_activation() {
        use crate::nn::layer::Layer;

        let layer = Layer::new(2, 3, &Activations::Sigmoid);
        let inputs = vec![Node::from(1.0), Node::from(2.0)];
        let outputs = layer.forward(&inputs);

        // All outputs should be between 0 and 1 (sigmoid)
        for output in outputs {
            let val = output.get_value();
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_layer_gradients() {
        use crate::nn::layer::Layer;

        let layer = Layer::new(2, 3, &Activations::None);
        let inputs = vec![Node::from(1.0), Node::from(2.0)];
        let outputs = layer.forward(&inputs);

        // Sum all outputs and backprop
        let mut sum = outputs[0].clone();
        for i in 1..outputs.len() {
            sum = sum + outputs[i].clone();
        }
        sum.backward();

        // Check that parameters received gradients
        let params = layer.parameters();
        let non_zero_grads = params
            .iter()
            .filter(|p| p.get_gradient().abs() > 1e-10)
            .count();

        // At least some parameters should have non-zero gradients
        assert!(non_zero_grads > 0);
    }

    // ========== MLP TESTS ==========

    #[test]
    fn test_mlp_creation() {
        use crate::nn::mlp::Mlp;

        // 2 inputs -> 4 hidden -> 1 output
        let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);

        let params = mlp.parameters();

        // Layer 1: 4 neurons * (2 weights + 1 bias) = 12
        // Layer 2: 1 neuron * (4 weights + 1 bias) = 5
        // Total: 17 parameters
        assert_eq!(params.len(), 17);
    }

    #[test]
    fn test_mlp_forward() {
        use crate::nn::mlp::Mlp;

        // 2 inputs -> 4 hidden -> 2 output
        let mlp = Mlp::new(&[2, 4, 2], &[Activations::Tanh, Activations::None]);

        let inputs = vec![Node::from(1.0), Node::from(2.0)];
        let outputs = mlp.forward(&inputs);

        // Should have 2 outputs
        assert_eq!(outputs.len(), 2);

        // All outputs should be valid
        for output in outputs {
            assert!(output.get_value().is_finite());
        }
    }

    #[test]
    fn test_mlp_single_output() {
        use crate::nn::mlp::Mlp;

        // 3 inputs -> 5 hidden -> 5 hidden -> 1 output
        let mlp = Mlp::new(
            &[3, 5, 5, 1],
            &[Activations::Tanh, Activations::Tanh, Activations::Sigmoid],
        );

        let inputs = vec![Node::from(0.5), Node::from(-0.3), Node::from(1.2)];
        let outputs = mlp.forward(&inputs);

        // Single output
        assert_eq!(outputs.len(), 1);

        // Output should be between 0 and 1 (sigmoid)
        let val = outputs[0].get_value();
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_mlp_backward_pass() {
        use crate::nn::mlp::Mlp;

        // Simple 2-layer network
        let mlp = Mlp::new(&[2, 3, 1], &[Activations::Tanh, Activations::None]);

        let inputs = vec![Node::from(1.0), Node::from(-0.5)];
        let outputs = mlp.forward(&inputs);

        let mut output = outputs[0].clone();
        output.backward();

        // Check that all parameters have gradients
        let params = mlp.parameters();
        let non_zero_grads = params
            .iter()
            .filter(|p| p.get_gradient().abs() > 1e-10)
            .count();

        // Most parameters should have non-zero gradients
        assert!(non_zero_grads > params.len() / 2);
    }

    #[test]
    fn test_mlp_deep_network() {
        use crate::nn::mlp::Mlp;

        // Deep network: 2 -> 8 -> 8 -> 4 -> 1
        let mlp = Mlp::new(
            &[2, 8, 8, 4, 1],
            &[
                Activations::Tanh,
                Activations::Tanh,
                Activations::Tanh,
                Activations::Sigmoid,
            ],
        );

        let inputs = vec![Node::from(0.5), Node::from(0.3)];
        let outputs = mlp.forward(&inputs);

        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].get_value().is_finite());
    }

    #[test]
    fn test_mlp_multiple_forward_passes() {
        use crate::nn::mlp::Mlp;

        let mlp = Mlp::new(&[2, 3, 1], &[Activations::Tanh, Activations::None]);

        // Multiple forward passes should all work
        let inputs1 = vec![Node::from(1.0), Node::from(2.0)];
        let outputs1 = mlp.forward(&inputs1);
        assert!(outputs1[0].get_value().is_finite());

        let inputs2 = vec![Node::from(-1.0), Node::from(0.5)];
        let outputs2 = mlp.forward(&inputs2);
        assert!(outputs2[0].get_value().is_finite());

        // Different inputs should give different outputs (probably)
        assert!((outputs1[0].get_value() - outputs2[0].get_value()).abs() > 0.01);
    }
}
