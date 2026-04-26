//! Tests for the autograd engine backpropagation correctness.

#[cfg(test)]
mod tests {
    use crate::engine::value::Node;

    const EPSILON: f64 = 1e-6;

    fn assert_close(actual: f64, expected: f64, msg: &str) {
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

        let e = std::f64::consts::E;
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

        let e2 = (2.0_f64).exp();
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
        let expected_val = 0.5_f64.tanh();
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
