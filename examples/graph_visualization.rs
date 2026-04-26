/// Example: Graph Visualization
///
/// This example demonstrates how to visualize computation graphs
/// in different formats (PNG, SVG, PDF).
use rusty_axon::engine::Node;

fn main() {
    println!("=== Graph Visualization Examples ===\n");

    // Example 1: Simple expression
    println!("1. Visualizing: (a + b) * c");
    let a = Node::from(2.0);
    let b = Node::from(3.0);
    let c = Node::from(4.0);
    let mut result = (a.clone() + b.clone()) * c.clone();
    result.backward();

    // Method 1: Save as DOT file only
    result.save_graph("simple_graph.dot").unwrap();

    // Method 2: Render to PNG (requires Graphviz)
    result.render_png("simple_graph").unwrap();

    // Method 3: Render to SVG (vector graphics - scalable)
    result.render_svg("simple_svg").unwrap();

    // Method 4: Render to PDF
    result.render_pdf("simple_pdf").unwrap();

    // Method 5: Generic render with custom format
    result.render_to("simple_custom", "jpg").unwrap();

    // Example 2: Complex neural network computation
    println!("\n2. Visualizing: Neural network computation");
    let x1 = Node::from(1.0);
    let x2 = Node::from(2.0);
    let w1 = Node::from(0.5);
    let w2 = Node::from(-0.3);
    let b = Node::from(1.0);

    let z = (w1 * x1) + (w2 * x2) + b;
    let mut activation = 1.0 / (1.0 + (-z).exp()); // Sigmoid
    activation.backward();

    activation.render_svg("neural_network").unwrap();

    // Example 3: Print DOT to console
    println!("\n3. DOT representation:");
    println!("{}", activation.to_dot());

    // Check Graphviz installation
    println!("\n4. System check:");
    if Node::check_graphviz() {
        println!("[+] Graphviz is installed and working!");
    } else {
        println!("[-] Graphviz not found.");
        println!("  Install it to generate images automatically.");
        println!("  For now, you can view .dot files at: http://www.webgraphviz.com/");
    }

    println!("\n Done!");
}
