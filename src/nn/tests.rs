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

    // -----------------------------------------------------------------
    // Phase 2 — Linear layer + ParamView
    // -----------------------------------------------------------------

    use crate::engine::matmul::ParamKind;
    use crate::nn::linear::Linear;
    use crate::optim::meprop::MeProp;
    use crate::optim::optimizer::Optimizer;
    use crate::optim::sgd::Sgd;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn test_linear_parameters_count() {
        // Acceptance: parameters().len() == in*out + out.
        let l = Linear::new(5, 3, Activations::None);
        assert_eq!(l.parameters().len(), 5 * 3 + 3);
    }

    #[test]
    fn test_linear_param_view_routes_to_tape() {
        // Mutating a param Node via set_value must show up in the tape's flat
        // weights buffer (since `matrixmultiply::sgemm` reads the buffer).
        let l = Linear::with_weights(
            2,
            2,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.5, -0.5],
            Activations::None,
        );
        let params = l.parameters();
        // First out*in entries are weights row-major; the last out are biases.
        assert!(close(params[0].get_value(), 1.0, 1e-6));
        assert!(close(params[3].get_value(), 4.0, 1e-6));
        assert!(close(params[4].get_value(), 0.5, 1e-6));

        let mut p0 = params[0].clone();
        p0.set_value(99.0);
        assert!(close(l.weights()[0], 99.0, 1e-6));
    }

    #[test]
    fn test_linear_forward_matches_manual_dot() {
        let l = Linear::with_weights(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![0.5, -0.5],
            Activations::None,
        );
        let inputs = vec![Node::from(1.0), Node::from(2.0), Node::from(3.0)];
        let outs = l.forward(&inputs);
        // y[0] = 1*1 + 2*2 + 3*3 + 0.5 = 14.5
        // y[1] = 4*1 + 5*2 + 6*3 - 0.5 = 31.5
        assert!(close(outs[0].get_value(), 14.5, 1e-5));
        assert!(close(outs[1].get_value(), 31.5, 1e-5));
    }

    #[test]
    fn test_linear_trains_y_eq_2x1_plus_3x2_plus_1() {
        // Acceptance: train Linear(2, 1, None) for 1000 steps on y = 2x1 + 3x2 + 1
        // and recover weights within 1% and bias within 1%.
        let l = Linear::with_weights(2, 1, vec![0.0, 0.0], vec![0.0], Activations::None);
        let mut opt = Sgd::new(0.05, l.parameters());

        // Deterministic LCG so the test does not depend on `rand`.
        let mut seed: u32 = 0xDEADBEEF;
        let next = |s: &mut u32| -> f32 {
            *s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((*s >> 8) as f32 / ((1u32 << 23) as f32)) - 1.0
        };

        for _ in 0..2000 {
            opt.zero_state();
            let x1 = next(&mut seed);
            let x2 = next(&mut seed);
            let target = 2.0 * x1 + 3.0 * x2 + 1.0;
            let outs = l.forward(&[Node::from(x1), Node::from(x2)]);
            // Squared-error loss with respect to target.
            let diff = outs[0].clone() - Node::from(target);
            let mut loss = diff.clone() * diff;
            loss.backward();
            opt.step();
        }

        let w = l.weights();
        let b = l.bias();
        // Weights are row-major [out, in]: row 0 = [w_x1, w_x2].
        assert!(close(w[0], 2.0, 0.05), "w_x1 = {}", w[0]);
        assert!(close(w[1], 3.0, 0.05), "w_x2 = {}", w[1]);
        assert!(close(b[0], 1.0, 0.05), "bias = {}", b[0]);
    }

    #[test]
    fn test_linear_meprop_top_k_selects_largest_grads() {
        // Build a Linear(8, 4) with frozen inputs that produce known gradients
        // per parameter under a sum-of-outputs loss; then verify MeProp at
        // top_k = 0.25 updates only the 25% (rounded up) parameters with the
        // largest |grad|.
        let in_dim = 8;
        let out_dim = 4;
        let total = in_dim * out_dim + out_dim; // 36

        let weights: Vec<f32> = (0..in_dim * out_dim).map(|i| 0.01 * i as f32).collect();
        let bias: Vec<f32> = vec![0.0; out_dim];
        let l = Linear::with_weights(
            in_dim,
            out_dim,
            weights.clone(),
            bias.clone(),
            Activations::None,
        );

        // Pick distinct large input values so dW[i, j] = x[j] differs across j
        // and ranks the gradients clearly.  loss = sum(y) => dL/dW[i,j] = x[j].
        let x_vals: Vec<f32> = (0..in_dim).map(|j| (j as f32 + 1.0) * 1.5).collect();
        let inputs: Vec<Node> = x_vals.iter().map(|&v| Node::from(v)).collect();

        let outs = l.forward(&inputs);
        let mut loss = outs[0].clone();
        for o in outs.iter().skip(1) {
            loss = loss + o.clone();
        }
        loss.backward();

        // Snapshot parameter values + gradients, then run MeProp and observe
        // which parameters changed.
        let params_before: Vec<f32> = l.parameters().iter().map(|p| p.get_value()).collect();
        let grads: Vec<f32> = l.parameters().iter().map(|p| p.get_gradient()).collect();
        assert_eq!(params_before.len(), total);

        let top_k = 0.25_f32;
        let mut opt = MeProp::new(0.1, l.parameters(), top_k);
        opt.step();

        // Expected: parameters whose |grad| is in the top ceil(0.25 * 36) = 9
        // updated; everyone else unchanged.
        let mut indexed: Vec<(usize, f32)> = grads.iter().map(|g| g.abs()).enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let k = (top_k * total as f32).ceil() as usize;
        let updated_indices: std::collections::HashSet<usize> =
            indexed.iter().take(k).map(|(i, _)| *i).collect();

        let params_after: Vec<f32> = l.parameters().iter().map(|p| p.get_value()).collect();
        for i in 0..total {
            if updated_indices.contains(&i) {
                assert!(
                    !close(params_after[i], params_before[i], 1e-9),
                    "param {} should have been updated",
                    i
                );
            } else {
                assert!(
                    close(params_after[i], params_before[i], 1e-9),
                    "param {} should be unchanged but moved by {}",
                    i,
                    params_after[i] - params_before[i]
                );
            }
        }
    }

    #[test]
    fn test_sgd_zero_state_dedups_param_tape() {
        // After backward(), the tape's d_weights are non-zero.  Sgd::zero_state
        // must call MatMulTape::reset_grads exactly once even though every
        // parameter Node references the same tape.
        let l = Linear::with_weights(
            2,
            2,
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0],
            Activations::None,
        );
        let mut opt = Sgd::new(0.01, l.parameters());

        let outs = l.forward(&[Node::from(1.0), Node::from(2.0)]);
        let mut loss = outs[0].clone() + outs[1].clone();
        loss.backward();

        // Sanity: gradients are populated.
        assert!(l.tape().d_weights_ref().iter().any(|&v| v != 0.0));
        assert!(l.tape().d_bias_ref().iter().any(|&v| v != 0.0));

        opt.zero_state();

        assert!(l.tape().d_weights_ref().iter().all(|&v| v == 0.0));
        assert!(l.tape().d_bias_ref().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_node_param_eq_and_hash() {
        // Two ParamView Nodes pointing at the same tape/kind/index must
        // compare equal and hash to the same bucket so the topo HashSet
        // dedupes them.
        use std::collections::HashSet;

        let l = Linear::new(3, 2, Activations::None);
        let p1 = l.parameters();
        let p2 = l.parameters();

        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a, b);
        }

        let mut set: HashSet<Node> = HashSet::new();
        for n in p1.iter() {
            set.insert(n.clone());
        }
        for n in p2.iter() {
            assert!(set.contains(n));
        }
        assert_eq!(set.len(), p1.len());
    }

    #[test]
    fn test_param_view_kinds_distinct() {
        let l = Linear::new(2, 2, Activations::None);
        let params = l.parameters();
        // Index 0 is a weight; index 4 is a bias.  Even if the underlying
        // index field were equal, the kind discriminates them.
        assert_ne!(params[0], params[4]);
        assert_eq!(ParamKind::Weight, ParamKind::Weight);
        assert_ne!(ParamKind::Weight, ParamKind::Bias);
    }

    #[test]
    fn test_linear_with_activation_relu() {
        // Sanity: activation chain is composed on top of the matmul outputs
        // and the gradient still flows through the fused matmul.
        let l = Linear::with_weights(
            2,
            1,
            vec![1.0, -1.0],
            vec![0.0],
            Activations::ReLU,
        );

        // x = (2, 1) -> raw = 2 - 1 = 1 -> relu = 1.  dL/dy = 1 (from sum).
        let mut out = l.forward(&[Node::from(2.0), Node::from(1.0)])
            .into_iter()
            .next()
            .unwrap();
        assert!(close(out.get_value(), 1.0, 1e-6));
        out.backward();
        // dW = d_out * x = [1*2, 1*1] = [2, 1]
        let dw = l.tape().d_weights_ref();
        assert!(close(dw[0], 2.0, 1e-5));
        assert!(close(dw[1], 1.0, 1e-5));
    }
}
