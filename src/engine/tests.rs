//! Tests for the autograd engine backpropagation correctness.

#[cfg(test)]
mod tests {
    use crate::engine::ops::Operation;
    use crate::engine::value::Node;

    // Engine is f32 end-to-end after Phase 0.5; ~1e-5 is the practical
    // accuracy of f32 accumulation over the small graphs exercised here.
    const EPSILON: f32 = 1e-5;

    fn assert_close(actual: f32, expected: f32, msg: &str) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "{}: expected {}, got {} (diff: {})",
            msg,
            expected,
            actual,
            (actual - expected).abs()
        );
    }

    #[test]
    fn test_simple_add() {
        // f(x, y) = x + y
        // df/dx = 1, df/dy = 1
        let x = Node::from(2.0);
        let y = Node::from(3.0);
        let mut f = x.clone() + y.clone();
        f.backward();

        assert_close(f.get_value(), 5.0, "forward value");
        assert_close(x.get_gradient(), 1.0, "df/dx");
        assert_close(y.get_gradient(), 1.0, "df/dy");
    }

    #[test]
    fn test_simple_mul() {
        // f(x, y) = x * y
        // df/dx = y, df/dy = x
        let x = Node::from(3.0);
        let y = Node::from(4.0);
        let mut f = x.clone() * y.clone();
        f.backward();

        assert_close(f.get_value(), 12.0, "forward value");
        assert_close(x.get_gradient(), 4.0, "df/dx");
        assert_close(y.get_gradient(), 3.0, "df/dy");
    }

    #[test]
    fn test_simple_sub() {
        // f(x, y) = x - y
        // df/dx = 1, df/dy = -1
        let x = Node::from(5.0);
        let y = Node::from(3.0);
        let mut f = x.clone() - y.clone();
        f.backward();

        assert_close(f.get_value(), 2.0, "forward value");
        assert_close(x.get_gradient(), 1.0, "df/dx");
        assert_close(y.get_gradient(), -1.0, "df/dy");
    }

    #[test]
    fn test_simple_div() {
        // f(x, y) = x / y
        // df/dx = 1/y, df/dy = -x/y²
        let x = Node::from(6.0);
        let y = Node::from(2.0);
        let mut f = x.clone() / y.clone();
        f.backward();

        assert_close(f.get_value(), 3.0, "forward value");
        assert_close(x.get_gradient(), 0.5, "df/dx");
        assert_close(y.get_gradient(), -1.5, "df/dy");
    }

    #[test]
    fn test_power() {
        // f(x) = x²
        // df/dx = 2x
        let x = Node::from(3.0);
        let mut f = x.pow(2.0);
        f.backward();

        assert_close(f.get_value(), 9.0, "forward value");
        assert_close(x.get_gradient(), 6.0, "df/dx");
    }

    #[test]
    fn test_power_cubic() {
        // f(x) = x³
        // df/dx = 3x²
        let x = Node::from(2.0);
        let mut f = x.pow(3.0);
        f.backward();

        assert_close(f.get_value(), 8.0, "forward value");
        assert_close(x.get_gradient(), 12.0, "df/dx");
    }

    #[test]
    fn test_exp() {
        // f(x) = e^x
        // df/dx = e^x
        let x = Node::from(1.0);
        let mut f = x.exp();
        f.backward();

        let e = std::f32::consts::E;
        assert_close(f.get_value(), e, "forward value");
        assert_close(x.get_gradient(), e, "df/dx");
    }

    #[test]
    fn test_relu_positive() {
        // f(x) = ReLU(x) where x > 0
        // df/dx = 1 when x > 0
        let x = Node::from(3.0);
        let mut f = x.relu();
        f.backward();

        assert_close(f.get_value(), 3.0, "forward value");
        assert_close(x.get_gradient(), 1.0, "df/dx");
    }

    #[test]
    fn test_relu_negative() {
        // f(x) = ReLU(x) where x < 0
        // df/dx = 0 when x < 0
        let x = Node::from(-3.0);
        let mut f = x.relu();
        f.backward();

        assert_close(f.get_value(), 0.0, "forward value");
        assert_close(x.get_gradient(), 0.0, "df/dx");
    }

    #[test]
    fn test_relu_zero() {
        // f(x) = ReLU(x) where x = 0
        // df/dx = 0 at x = 0 (by convention)
        let x = Node::from(0.0);
        let mut f = x.relu();
        f.backward();

        assert_close(f.get_value(), 0.0, "forward value");
        assert_close(x.get_gradient(), 0.0, "df/dx");
    }

    #[test]
    fn test_relu_chain_rule() {
        // f(x) = ReLU(2x + 1)
        // For x = 1: 2x + 1 = 3 > 0, so ReLU(3) = 3
        // df/dx = 1 * 2 = 2 (chain rule)
        let x = Node::from(1.0);
        let inner = x.clone() * 2.0 + 1.0;
        let mut f = inner.relu();
        f.backward();

        assert_close(f.get_value(), 3.0, "forward value");
        assert_close(x.get_gradient(), 2.0, "df/dx");
    }

    #[test]
    fn test_relu_chain_rule_negative() {
        // f(x) = ReLU(2x + 1)
        // For x = -2: 2x + 1 = -3 < 0, so ReLU(-3) = 0
        // df/dx = 0 * 2 = 0 (gradient killed)
        let x = Node::from(-2.0);
        let inner = x.clone() * 2.0 + 1.0;
        let mut f = inner.relu();
        f.backward();

        assert_close(f.get_value(), 0.0, "forward value");
        assert_close(x.get_gradient(), 0.0, "df/dx");
    }

    #[test]
    fn test_chain_rule_simple() {
        // f(x) = (x + 1)²
        // df/dx = 2(x + 1) = 2x + 2
        let x = Node::from(2.0);
        let temp = x.clone() + 1.0;
        let mut f = temp.pow(2.0);
        f.backward();

        assert_close(f.get_value(), 9.0, "forward value");
        assert_close(x.get_gradient(), 6.0, "df/dx");
    }

    #[test]
    fn test_chain_rule_product() {
        // f(x, y) = (x * y)²
        // df/dx = 2xy * y = 2xy²
        // df/dy = 2xy * x = 2x²y
        let x = Node::from(3.0);
        let y = Node::from(4.0);
        let temp = x.clone() * y.clone();
        let mut f = temp.pow(2.0);
        f.backward();

        assert_close(f.get_value(), 144.0, "forward value");
        assert_close(x.get_gradient(), 96.0, "df/dx"); // 2*3*16 = 96
        assert_close(y.get_gradient(), 72.0, "df/dy"); // 2*9*4 = 72
    }

    #[test]
    fn test_multiple_uses_of_same_variable() {
        // f(x) = x * x (different from x²)
        // df/dx = x + x = 2x
        let x = Node::from(3.0);
        let mut f = x.clone() * x.clone();
        f.backward();

        assert_close(f.get_value(), 9.0, "forward value");
        assert_close(x.get_gradient(), 6.0, "df/dx");
    }

    #[test]
    fn test_multiple_paths_complex() {
        // f(x) = x + x * x
        // df/dx = 1 + 2x
        let x = Node::from(3.0);
        let x_squared = x.clone() * x.clone();
        let mut f = x.clone() + x_squared;
        f.backward();

        assert_close(f.get_value(), 12.0, "forward value");
        assert_close(x.get_gradient(), 7.0, "df/dx"); // 1 + 2*3 = 7
    }

    #[test]
    fn test_complex_expression() {
        // f(x, y) = (x + y) * (x - y)
        // = x² - y²
        // df/dx = 2x, df/dy = -2y
        let x = Node::from(5.0);
        let y = Node::from(3.0);
        let sum = x.clone() + y.clone();
        let diff = x.clone() - y.clone();
        let mut f = sum * diff;
        f.backward();

        assert_close(f.get_value(), 16.0, "forward value");
        assert_close(x.get_gradient(), 10.0, "df/dx");
        assert_close(y.get_gradient(), -6.0, "df/dy");
    }

    #[test]
    fn test_division_chain() {
        // f(x, y) = x / (y + 1)
        // df/dx = 1/(y+1)
        // df/dy = -x/(y+1)²
        let x = Node::from(6.0);
        let y = Node::from(2.0);
        let denom = y.clone() + 1.0;
        let mut f = x.clone() / denom;
        f.backward();

        assert_close(f.get_value(), 2.0, "forward value");
        assert_close(x.get_gradient(), 1.0 / 3.0, "df/dx");
        // df/dy = -x/(y+1)² = -6/9 = -2/3
        assert_close(y.get_gradient(), -2.0 / 3.0, "df/dy");
    }

    #[test]
    fn test_exp_chain() {
        // f(x) = e^(2x)
        // df/dx = 2e^(2x)
        let x = Node::from(1.0);
        let two_x = x.clone() * 2.0;
        let mut f = two_x.exp();
        f.backward();

        let e2 = (2.0_f32).exp();
        assert_close(f.get_value(), e2, "forward value");
        assert_close(x.get_gradient(), 2.0 * e2, "df/dx");
    }

    #[test]
    fn test_tanh_approximation() {
        // tanh(x) ≈ (e^(2x) - 1) / (e^(2x) + 1)
        // This tests the complex expression from main.rs
        let x = Node::from(0.5);
        let two_x = x.clone() * 2.0;
        let exp_2x = two_x.exp();
        let numerator = exp_2x.clone() - 1.0;
        let denominator = exp_2x.clone() + 1.0;
        let mut f = numerator / denominator;
        f.backward();

        // tanh(0.5) ≈ 0.46211715726
        let expected_val = 0.5_f32.tanh();
        assert_close(f.get_value(), expected_val, "forward value");

        // d/dx tanh(x) = 1 - tanh²(x)
        let expected_grad = 1.0 - expected_val * expected_val;
        assert_close(x.get_gradient(), expected_grad, "df/dx");
    }

    #[test]
    fn test_sigmoid_approximation() {
        // sigmoid(x) ≈ 1 / (1 + e^(-x))
        let x = Node::from(0);

        let sigmoid = 1.0 / (1.0 + (-x.clone()).exp());
        let mut f = sigmoid;
        f.backward();
        assert_close(f.get_value(), 0.5, "forward value");
        assert_close(x.get_gradient(), 0.25, "df/dx");
    }
    #[test]
    fn test_neuron_like_computation() {
        // Simulate: output = (w1*x1 + w2*x2 + b)²
        let x1 = Node::from(2.0);
        let x2 = Node::from(3.0);
        let w1 = Node::from(0.5);
        let w2 = Node::from(-0.3);
        let b = Node::from(1.0);

        let term1 = w1.clone() * x1.clone();
        let term2 = w2.clone() * x2.clone();
        let sum = term1 + term2 + b.clone();
        let mut output = sum.pow(2.0);
        output.backward();

        // Forward: (0.5*2 + (-0.3)*3 + 1)² = (1.0 - 0.9 + 1.0)² = 1.1² = 1.21
        assert_close(output.get_value(), 1.21, "forward value");

        // Check gradients make sense (non-zero and reasonable magnitudes)
        assert!(x1.get_gradient().abs() > 0.0);
        assert!(x2.get_gradient().abs() > 0.0);
        assert!(w1.get_gradient().abs() > 0.0);
        assert!(w2.get_gradient().abs() > 0.0);
        assert!(b.get_gradient().abs() > 0.0);
    }

    #[test]
    fn test_scalar_operations() {
        // Test Node * scalar operations
        let x = Node::from(3.0);
        let mut f = x.clone() * 2.0;
        f.backward();

        assert_close(f.get_value(), 6.0, "forward value");
        assert_close(x.get_gradient(), 2.0, "df/dx");
    }

    #[test]
    fn test_negative_values() {
        // f(x, y) = x * y where x < 0
        let x = Node::from(-2.0);
        let y = Node::from(3.0);
        let mut f = x.clone() * y.clone();
        f.backward();

        assert_close(f.get_value(), -6.0, "forward value");
        assert_close(x.get_gradient(), 3.0, "df/dx");
        assert_close(y.get_gradient(), -2.0, "df/dy");
    }

    #[test]
    fn test_zero_gradient_isolation() {
        // f(x, y, z) = x * y
        // df/dz should be 0 (z is not used)
        let x = Node::from(2.0);
        let y = Node::from(3.0);
        let z = Node::from(5.0);
        let mut f = x.clone() * y.clone();
        f.backward();

        assert_close(z.get_gradient(), 0.0, "df/dz should be 0");
    }

    #[test]
    fn test_long_chain() {
        // f(x) = ((x + 1) * 2)² - 3
        // Test a longer chain of operations
        let x = Node::from(2.0);
        let step1 = x.clone() + 1.0; // 3
        let step2 = step1 * 2.0; // 6
        let step3 = step2.pow(2.0); // 36
        let mut f = step3 - 3.0; // 33
        f.backward();

        assert_close(f.get_value(), 33.0, "forward value");

        // df/dx using chain rule:
        // d/dx[((x+1)*2)² - 3] = 2*((x+1)*2) * 2 * 1 = 4(x+1)*2 = 8(x+1) = 24
        assert_close(x.get_gradient(), 24.0, "df/dx");
    }

    #[test]
    fn test_division_by_itself() {
        // f(x) = x / x = 1
        // df/dx = (x - x) / x² = 0
        let x = Node::from(5.0);
        let mut f = x.clone() / x.clone();
        f.backward();

        assert_close(f.get_value(), 1.0, "forward value");
        assert_close(x.get_gradient(), 0.0, "df/dx");
    }

    #[test]
    fn test_power_fractional() {
        // f(x) = x^0.5 (square root)
        // df/dx = 0.5 * x^(-0.5) = 0.5 / sqrt(x)
        let x = Node::from(4.0);
        let mut f = x.pow(0.5);
        f.backward();

        assert_close(f.get_value(), 2.0, "forward value");
        assert_close(x.get_gradient(), 0.25, "df/dx"); // 0.5 / 2 = 0.25
    }

    #[test]
    fn test_graph_visualization() {
        // Test that to_dot generates valid DOT format
        let a = Node::from(2.0);
        let b = Node::from(3.0);
        let mut c = a.clone() + b.clone();
        c.backward();

        let dot = c.to_dot();

        // Check basic DOT structure
        assert!(dot.contains("digraph G"));
        assert!(dot.contains("rankdir=LR"));

        // Check that nodes are present
        assert!(dot.contains("val=5.0000"));
        assert!(dot.contains("val=2.0000"));
        assert!(dot.contains("val=3.0000"));

        // Check that gradients are present
        assert!(dot.contains("grad=1.0000"));

        // Check that operation is present
        assert!(dot.contains("label=\"+\""));
    }

    #[test]
    fn test_value_struct_is_f32_packed() {
        // Phase 0.5 acceptance: value (f32) + gradient (f32) + Operation enum.
        // The two scalar fields must occupy 8 bytes total — half of the f64
        // engine — and Operation is the enum tail.
        use crate::engine::value::Value;
        // Sanity: each scalar field is 4 bytes, not 8.
        assert_eq!(std::mem::size_of::<f32>(), 4);
        // The two scalar fields together are 8 bytes; Operation adds its
        // discriminant + payload after them. We only assert the scalar
        // contribution (the 8 B reduction is the paper-relevant figure).
        assert!(
            std::mem::size_of::<Value>() < std::mem::size_of::<(f64, f64)>()
                + std::mem::size_of::<crate::engine::ops::Operation>(),
            "Value should be smaller than the f64 layout would have been"
        );
    }

    // ---------------------------------------------------------------------
    // Phase 1 — fused MatMul op + MatMulTape
    // ---------------------------------------------------------------------

    /// Reference scalar matmul, for cross-checking the fused tape.
    fn naive_matvec(weights: &[f32], bias: &[f32], input: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = input.len();
        (0..out_dim)
            .map(|i| {
                let row = i * in_dim;
                let dot: f32 = (0..in_dim).map(|j| weights[row + j] * input[j]).sum();
                bias[i] + dot
            })
            .collect()
    }

    fn linear_congruential(seed: &mut u32) -> f32 {
        // A tiny LCG so tests are deterministic without pulling rand into
        // engine/tests.  Maps to roughly [-1.0, 1.0).
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((*seed >> 8) as f32 / ((1u32 << 23) as f32)) - 1.0
    }

    #[test]
    fn test_matmul_forward_simple() {
        // 2x3 weights, identity check on the forward kernel.
        use crate::engine::matmul::MatMulTape;

        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3] row-major
        let bias = vec![0.5, -0.5];
        let tape = MatMulTape::new(3, 2, weights.clone(), bias.clone());

        let input_nodes: Vec<Node> = vec![Node::from(1.0), Node::from(2.0), Node::from(3.0)];
        let outputs = tape.forward(&input_nodes);

        assert_eq!(outputs.len(), 2);
        // y[0] = 1*1 + 2*2 + 3*3 + 0.5 = 14.5
        // y[1] = 4*1 + 5*2 + 6*3 - 0.5 = 31.5
        assert_close(outputs[0].get_value(), 14.5, "y[0]");
        assert_close(outputs[1].get_value(), 31.5, "y[1]");
    }

    #[test]
    fn test_matmul_backward_leaves_8x4() {
        // 8x4 weight, [4] input that are pure leaves.  Compare gradients
        // against a hand-computed reference.
        use crate::engine::matmul::MatMulTape;

        let mut seed = 0xC0FFEE_u32;
        let in_dim = 4;
        let out_dim = 8;
        let weights: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| linear_congruential(&mut seed))
            .collect();
        let bias: Vec<f32> = (0..out_dim)
            .map(|_| linear_congruential(&mut seed))
            .collect();
        let input_vals: Vec<f32> = (0..in_dim)
            .map(|_| linear_congruential(&mut seed))
            .collect();

        let tape = MatMulTape::new(in_dim, out_dim, weights.clone(), bias.clone());
        let inputs: Vec<Node> = input_vals.iter().map(|&v| Node::from(v)).collect();
        let outputs = tape.forward(&inputs);

        // Forward sanity.
        let expected_y = naive_matvec(&weights, &bias, &input_vals, out_dim);
        for i in 0..out_dim {
            assert_close(outputs[i].get_value(), expected_y[i], &format!("y[{}]", i));
        }

        // Use loss = sum(y) so dL/dy[i] = 1.  Then dL/dW[i,j] = x[j],
        // dL/db[i] = 1, dL/dx[j] = sum_i W[i,j].
        let mut loss = outputs[0].clone();
        for o in outputs.iter().skip(1) {
            loss = loss + o.clone();
        }
        loss.backward();

        let dw = tape.d_weights_ref();
        let db = tape.d_bias_ref();
        for i in 0..out_dim {
            for j in 0..in_dim {
                assert_close(
                    dw[i * in_dim + j],
                    input_vals[j],
                    &format!("dW[{},{}]", i, j),
                );
            }
            assert_close(db[i], 1.0, &format!("db[{}]", i));
        }

        // Inputs are leaves, so no upstream propagation is performed and the
        // input Nodes' gradients should remain zero.
        for (j, n) in inputs.iter().enumerate() {
            assert_close(n.get_gradient(), 0.0, &format!("leaf x[{}] grad", j));
        }
    }

    #[test]
    fn test_matmul_backward_chained_inputs() {
        // The matmul's input vector is *itself* produced by a small upstream
        // graph: x[j] = (a + b) * c  for distinct (a, b, c) per j.  Verifies
        // that `dx = Wᵀ d_out` is propagated into the upstream Nodes and that
        // the topo walk orders them correctly.
        use crate::engine::matmul::MatMulTape;

        let mut seed = 0xBADBEEF_u32;
        let in_dim = 3;
        let out_dim = 4;
        let weights: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| linear_congruential(&mut seed))
            .collect();
        let bias: Vec<f32> = (0..out_dim)
            .map(|_| linear_congruential(&mut seed))
            .collect();
        let tape = MatMulTape::new(in_dim, out_dim, weights.clone(), bias.clone());

        // Build chained upstream Nodes.
        let a: Vec<Node> = (0..in_dim).map(|i| Node::from(0.5 + i as f32)).collect();
        let b: Vec<Node> = (0..in_dim).map(|i| Node::from(-0.25 + i as f32)).collect();
        let c: Vec<Node> = (0..in_dim).map(|i| Node::from(2.0 - 0.3 * i as f32)).collect();
        let x: Vec<Node> = (0..in_dim)
            .map(|i| (a[i].clone() + b[i].clone()) * c[i].clone())
            .collect();
        let x_vals: Vec<f32> = x.iter().map(|n| n.get_value()).collect();

        let outputs = tape.forward(&x);

        // loss = sum(y), so dL/dy[i] = 1, dL/dx[j] = sum_i W[i,j].
        let mut loss = outputs[0].clone();
        for o in outputs.iter().skip(1) {
            loss = loss + o.clone();
        }
        loss.backward();

        // Reference dx[j] = sum_i W[i, j].
        let dx_ref: Vec<f32> = (0..in_dim)
            .map(|j| (0..out_dim).map(|i| weights[i * in_dim + j]).sum::<f32>())
            .collect();

        // d_input scratch matches dx_ref.
        let d_input = tape.d_input_ref();
        for j in 0..in_dim {
            assert_close(d_input[j], dx_ref[j], &format!("d_input[{}]", j));
        }

        // dx propagates into a[j], b[j], c[j] via the upstream chain.
        // x[j] = (a[j] + b[j]) * c[j]  =>  dL/da[j] = dx[j] * c[j],
        // dL/db[j] = dx[j] * c[j], dL/dc[j] = dx[j] * (a[j] + b[j]).
        for j in 0..in_dim {
            let cv = c[j].get_value();
            let ab = a[j].get_value() + b[j].get_value();
            assert_close(a[j].get_gradient(), dx_ref[j] * cv, &format!("dL/da[{}]", j));
            assert_close(b[j].get_gradient(), dx_ref[j] * cv, &format!("dL/db[{}]", j));
            assert_close(c[j].get_gradient(), dx_ref[j] * ab, &format!("dL/dc[{}]", j));
        }

        // Forward kernel correctness.
        let expected_y = naive_matvec(&weights, &bias, &x_vals, out_dim);
        for i in 0..out_dim {
            assert_close(outputs[i].get_value(), expected_y[i], &format!("y[{}]", i));
        }
    }

    #[test]
    fn test_matmul_d_weights_accumulate_across_backwards() {
        // d_weights should accumulate across multiple backward passes until
        // an explicit `reset_grads()` call.  This is the contract Phase 2's
        // optimizer relies on for mini-batch gradient accumulation.
        use crate::engine::matmul::MatMulTape;

        let weights = vec![0.5, -0.5, 1.0, 2.0]; // [2, 2]
        let bias = vec![0.0, 0.0];
        let tape = MatMulTape::new(2, 2, weights.clone(), bias.clone());

        for _ in 0..3 {
            let inputs = vec![Node::from(1.0), Node::from(1.0)];
            let outs = tape.forward(&inputs);
            let mut loss = outs[0].clone() + outs[1].clone();
            loss.backward();
        }

        // After 3 backwards on `loss = y0 + y1` with x = [1, 1]:
        //   dW[i, j] = 3 * x[j] = 3
        //   db[i] = 3
        let dw = tape.d_weights_ref();
        let db = tape.d_bias_ref();
        for v in dw.iter() {
            assert_close(*v, 3.0, "accumulated dW");
        }
        for v in db.iter() {
            assert_close(*v, 3.0, "accumulated db");
        }

        drop(dw);
        drop(db);
        tape.reset_grads();
        assert!(tape.d_weights_ref().iter().all(|&v| v == 0.0));
        assert!(tape.d_bias_ref().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_matmul_visit_count_resets_between_backwards() {
        // Two consecutive backwards on two separate forwards must each fire
        // `run_backward` exactly once, not piggy-back on the prior pass'
        // visit_count.
        use crate::engine::matmul::MatMulTape;

        let tape = MatMulTape::new(1, 2, vec![1.0, 2.0], vec![0.0, 0.0]);

        for _ in 0..2 {
            let inputs = vec![Node::from(1.0)];
            let outs = tape.forward(&inputs);
            assert_eq!(tape.visit_count.get(), 0, "forward resets visit_count");
            assert!(!tape.backward_done.get(), "forward resets backward_done");

            let mut loss = outs[0].clone() + outs[1].clone();
            loss.backward();
            assert!(tape.backward_done.get(), "run_backward fired");
            assert_eq!(tape.visit_count.get(), 2, "all outputs accumulated");
        }
    }

    #[test]
    fn test_matmul_topo_walked_resets_after_backward() {
        // After `backward()` returns, `topo_walked` must be `false` so the
        // next `backward()` call walks `upstream` correctly.
        use crate::engine::matmul::MatMulTape;

        let tape = MatMulTape::new(2, 1, vec![1.0, 1.0], vec![0.0]);
        let a = Node::from(1.0);
        let b = Node::from(2.0);
        let inputs = vec![a.clone() + Node::from(0.0), b.clone() + Node::from(0.0)];
        let outs = tape.forward(&inputs);

        let mut loss = outs[0].clone();
        loss.backward();
        assert!(
            !tape.topo_walked.get(),
            "topo_walked must be cleared after backward"
        );
    }

    #[test]
    fn test_operation_size_regression() {
        // Phase 1 acceptance: adding `MatMul { Rc<MatMulTape>, usize }` keeps
        // `Operation` well under the 64-byte ceiling the plan allows.
        // Two-Node variants (Add, Sub, Mul, Div) dominate at 16 B of payload;
        // MatMul is 16 B too (Rc + usize).  Total with discriminant fits in
        // ~24 B in practice.
        assert!(
            std::mem::size_of::<Operation>() <= 64,
            "Operation grew past the 64-byte ceiling: {}",
            std::mem::size_of::<Operation>()
        );
    }

    #[test]
    fn test_complex_graph_visualization() {
        // Test with more complex expression
        let x = Node::from(2.0);
        let y = Node::from(3.0);
        let z = (x.clone() * y.clone()).pow(2.0);
        let mut result = z / (x.clone() + 1.0);
        result.backward();

        let dot = result.to_dot();

        // Should contain multiple operations
        assert!(dot.contains("×")); // multiplication
        assert!(dot.contains("+")); // addition
        assert!(dot.contains("÷")); // division
        assert!(dot.contains("^")); // power
    }
}
