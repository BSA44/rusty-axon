/// Example: Basic Autograd Operations
/// 
/// Demonstrates the core automatic differentiation capabilities.
/// Run with: cargo run --example basic_autograd

use rusty_axon::engine::Node;

fn main() {
    println!("=== Basic Autograd Examples ===\n");

    // Example 1: Simple arithmetic
    println!("1. Simple expression: f = (a + b) * c");
    let a = Node::from(2.0);
    let b = Node::from(-3.0);
    let c = Node::from(10.0);
    
    let d = a.clone() * b.clone();
    let e = d + c.clone();
    let mut f = e.pow(2.0);
    
    println!("   Forward: f = {}", f.get_value());
    
    f.backward();
    
    println!("   Gradients:");
    println!("     df/da = {}", a.get_gradient());
    println!("     df/db = {}", b.get_gradient());
    println!("     df/dc = {}", c.get_gradient());

    // Example 2: Power rule
    println!("\n2. Power rule: y = x³");
    let x = Node::from(2.0);
    let mut y = x.clone().pow(3.0);
    y.backward();
    println!("   x = {}, y = x³ = {}", x.get_value(), y.get_value());
    println!("   dy/dx = 3x² = {}", x.get_gradient());

    // Example 3: Exponential and logarithm
    println!("\n3. Exp and Log: y = exp(x), z = ln(x)");
    let x = Node::from(1.0);
    let mut exp_x = x.clone().exp();
    exp_x.backward();
    println!("   exp(1) = {:.4}, d/dx exp(x) = {:.4}", exp_x.get_value(), x.get_gradient());

    let x2 = Node::from(std::f64::consts::E);
    let mut ln_x = x2.clone().log(std::f64::consts::E);
    ln_x.backward();
    println!("   ln(e) = {:.4}, d/dx ln(x) = {:.4}", ln_x.get_value(), x2.get_gradient());

    // Example 4: Sigmoid activation
    println!("\n4. Sigmoid: σ(x) = 1 / (1 + e^(-x))");
    let x = Node::from(0.0);
    let mut sigmoid = 1.0 / (1.0 + (-x.clone()).exp());
    sigmoid.backward();
    println!("   σ(0) = {}", sigmoid.get_value());
    println!("   σ'(0) = {} (expected: 0.25)", x.get_gradient());

    // Example 5: Multiple paths (gradient accumulation)
    println!("\n5. Multiple paths: f = x * x + x");
    let x = Node::from(3.0);
    let x_squared = x.clone() * x.clone();
    let mut f = x.clone() + x_squared;
    f.backward();
    println!("   f(3) = {}", f.get_value());
    println!("   df/dx = 2x + 1 = {}", x.get_gradient());

    println!("\n✨ All examples completed!");
}

