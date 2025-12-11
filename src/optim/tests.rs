#[cfg(test)]
mod tests {
    use crate::engine::Node;
    use crate::nn::mlp::Mlp;
    use crate::nn::activations::Activations;
    use crate::optim::optimizer::Optimizer;

    // ========== SGD TESTS ==========

    #[test]
    fn test_sgd_creation() {
        use crate::optim::sgd::Sgd;
        
        let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
        let _optimizer = Sgd::new(0.1, mlp.parameters());
    }

    #[test]
    fn test_sgd_step() {
        use crate::optim::sgd::Sgd;
        
        let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
        let params = mlp.parameters();
        let initial_values: Vec<f64> = params.iter().map(|p| p.get_value()).collect();
        
        let mut optimizer = Sgd::new(0.1, params);
        
        // Set some gradients
        for param in mlp.parameters().iter() {
            param.add_gradient(1.0);
        }
        
        // Step
        optimizer.step();
        
        // Values should have decreased (gradient was positive, lr positive)
        for (i, param) in mlp.parameters().iter().enumerate() {
            assert!(
                param.get_value() < initial_values[i],
                "Parameter {} should decrease: {} -> {}",
                i, initial_values[i], param.get_value()
            );
        }
    }

    #[test]
    fn test_sgd_zero_state() {
        use crate::optim::sgd::Sgd;
        
        let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
        let mut optimizer = Sgd::new(0.1, mlp.parameters());
        
        // Set some gradients
        for param in mlp.parameters().iter() {
            param.add_gradient(5.0);
        }
        
        // Verify gradients are set
        for param in mlp.parameters().iter() {
            assert!(param.get_gradient().abs() > 0.0);
        }
        
        // Zero state
        optimizer.zero_state();
        
        // All gradients should be zero
        for param in mlp.parameters().iter() {
            assert_eq!(param.get_gradient(), 0.0, "Gradient should be zero");
        }
    }

    // ========== PARALLEL SGD TESTS ==========

    #[test]
    fn test_parallel_sgd_creation() {
        use crate::optim::parallel_sgd::ParallelSgd;
        
        let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
        let optimizer = ParallelSgd::new(0.1, mlp.parameters());
        
        assert_eq!(optimizer.get_learning_rate(), 0.1);
        assert_eq!(optimizer.num_parameters(), 17); // (2+1)*4 + (4+1)*1 = 17
    }

    #[test]
    fn test_parallel_sgd_step() {
        use crate::optim::parallel_sgd::ParallelSgd;
        
        let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
        let params = mlp.parameters();
        let initial_values: Vec<f64> = params.iter().map(|p| p.get_value()).collect();
        
        let mut optimizer = ParallelSgd::new(0.1, params);
        
        // Set some gradients
        for param in mlp.parameters().iter() {
            param.add_gradient(1.0);
        }
        
        // Step
        optimizer.step();
        
        // Values should have decreased (gradient was positive)
        for (i, param) in mlp.parameters().iter().enumerate() {
            assert!(
                param.get_value() < initial_values[i],
                "Parameter {} should decrease", i
            );
        }
    }

    #[test]
    fn test_parallel_sgd_zero_state() {
        use crate::optim::parallel_sgd::ParallelSgd;
        
        let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
        let mut optimizer = ParallelSgd::new(0.1, mlp.parameters());
        
        // Set some gradients
        for param in mlp.parameters().iter() {
            param.add_gradient(5.0);
        }
        
        // Zero state
        optimizer.zero_state();
        
        // All gradients should be zero
        for param in mlp.parameters().iter() {
            assert_eq!(param.get_gradient(), 0.0, "Gradient should be zero");
        }
    }

    #[test]
    fn test_parallel_sgd_set_learning_rate() {
        use crate::optim::parallel_sgd::ParallelSgd;
        
        let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
        let mut optimizer = ParallelSgd::new(0.1, mlp.parameters());
        
        assert_eq!(optimizer.get_learning_rate(), 0.1);
        
        optimizer.set_learning_rate(0.01);
        assert_eq!(optimizer.get_learning_rate(), 0.01);
    }

    #[test]
    fn test_parallel_sgd_training_loop() {
        use crate::optim::parallel_sgd::ParallelSgd;
        
        let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
        let mut optimizer = ParallelSgd::new(0.5, mlp.parameters());
        
        // XOR training data
        let data = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ];
        
        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;
        
        for epoch in 0..100 {
            let mut epoch_loss = 0.0;
            
            for (inputs, target) in &data {
                optimizer.zero_state();
                
                let input_nodes: Vec<Node> = inputs.iter().map(|&x| Node::from(x)).collect();
                let outputs = mlp.forward(&input_nodes);
                
                let diff = outputs[0].clone() - Node::from(*target);
                let mut loss = diff.pow(2.0);
                epoch_loss += loss.get_value();
                
                loss.backward();
                optimizer.step();
            }
            
            if epoch == 0 {
                initial_loss = epoch_loss;
            }
            final_loss = epoch_loss;
        }
        
        assert!(final_loss < initial_loss, "Loss should decrease during training");
    }

    // ========== MEPROP TESTS ==========

    #[test]
    fn test_meprop_creation() {
        use crate::optim::meprop::MeProp;
        
        let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
        let _optimizer = MeProp::new(0.1, mlp.parameters(), 0.5);
    }

    #[test]
    fn test_meprop_sparse_update() {
        use crate::optim::meprop::MeProp;
        
        let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
        let params = mlp.parameters();
        let initial_values: Vec<f64> = params.iter().map(|p| p.get_value()).collect();
        
        // top_k = 0.5 means only 50% of parameters get updated
        let mut optimizer = MeProp::new(0.1, params, 0.5);
        
        // Set different gradients so we can see which ones get updated
        for (i, param) in mlp.parameters().iter().enumerate() {
            param.add_gradient((i + 1) as f64);
        }
        
        optimizer.step();
        
        // Count how many parameters changed
        let changed_count = mlp.parameters().iter()
            .zip(initial_values.iter())
            .filter(|(p, &init)| (p.get_value() - init).abs() > 1e-10)
            .count();
        
        // With top_k = 0.5 and 17 params, ceil(17 * 0.5) = 9 should be updated
        let expected = (17.0_f64 * 0.5).ceil() as usize;
        assert_eq!(changed_count, expected, "MeProp should update top {}% of parameters", 50);
    }
}

