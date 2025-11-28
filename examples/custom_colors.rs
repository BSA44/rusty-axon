//! Example demonstrating custom color schemes for neural network visualization.

use rusty_axon::nn::mlp::Mlp;
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::visualization::NetworkVisualizationConfig;

fn main() {
    println!("=== Custom Color Schemes for Neural Networks ===\n");

    // Create a simple network
    let mlp = Mlp::new(
        &[3, 6, 6, 2],
        &[Activations::Tanh, Activations::Tanh, Activations::Sigmoid]
    );

    // Example 1: Default colors
    println!("Example 1: Default Color Scheme");
    println!("  Colors: Blue (Input), Yellow (Hidden), Green (Output)");
    mlp.visualize_network("network_default", "png").unwrap();
    println!("  [+] Saved to network_default.png\n");

    // Example 2: Purple/Pink theme
    println!("Example 2: Purple/Pink Theme");
    let purple_config = NetworkVisualizationConfig::with_colors(
        "lavender", "mediumpurple",      // Input: light purple bg, medium purple neurons
        "mistyrose", "lightcoral",       // Hidden: light pink bg, coral neurons
        "lightcyan", "lightskyblue",     // Output: light cyan bg, sky blue neurons
    );
    mlp.visualize_network_with_config("network_purple", "png", &purple_config).unwrap();
    println!("  [+] Saved to network_purple.png\n");

    // Example 3: Warm theme (reds and oranges)
    println!("Example 3: Warm Theme (Red/Orange)");
    let warm_config = NetworkVisualizationConfig::with_colors(
        "peachpuff", "lightsalmon",      // Input: peach bg, salmon neurons
        "moccasin", "orange",             // Hidden: moccasin bg, orange neurons
        "mistyrose", "indianred",         // Output: misty rose bg, indian red neurons
    );

    let warm_config = warm_config.with_edges("red", 1.5);
    mlp.visualize_network_with_config("network_warm", "png", &warm_config).unwrap();
    println!("  [+] Saved to network_warm.png\n");

    // Example 4: Cool theme (blues and greens)
    println!("Example 4: Cool Theme (Blue/Green)");
    let cool_config = NetworkVisualizationConfig::with_colors(
        "lightcyan", "deepskyblue",      // Input: light cyan bg, deep sky blue neurons
        "paleturquoise", "mediumturquoise", // Hidden: pale turquoise bg, medium turquoise neurons
        "lightgreen", "seagreen",         // Output: light green bg, sea green neurons
    );
    mlp.visualize_network_with_config("network_cool", "png", &cool_config).unwrap();
    println!("  [+] Saved to network_cool.png\n");

    // Example 5: Monochrome (grayscale)
    println!("Example 5: Monochrome (Grayscale)");
    let mono_config = NetworkVisualizationConfig::with_colors(
        "whitesmoke", "lightgray",       // Input: whitesmoke bg, light gray neurons
        "gainsboro", "darkgray",          // Hidden: gainsboro bg, dark gray neurons
        "silver", "gray",                 // Output: silver bg, gray neurons
    );
    mlp.visualize_network_with_config("network_mono", "png", &mono_config).unwrap();
    println!("  [+] Saved to network_mono.png\n");

    // Example 6: High contrast (black background theme)
    println!("Example 6: Vibrant Theme");
    let vibrant_config = NetworkVisualizationConfig::with_colors(
        "lightpink", "hotpink",           // Input: light pink bg, hot pink neurons
        "khaki", "gold",                  // Hidden: khaki bg, gold neurons
        "palegreen", "limegreen",         // Output: pale green bg, lime green neurons
    );
    mlp.visualize_network_with_config("network_vibrant", "png", &vibrant_config).unwrap();
    println!("  [+] Saved to network_vibrant.png\n");

    // Example 7: Custom edge colors
    println!("Example 7: Custom Edge Colors");
    let edge_config = NetworkVisualizationConfig::with_colors(
        "aliceblue", "royalblue",
        "lavenderblush", "orchid",
        "honeydew", "mediumseagreen",
    ).with_edges("navy", 1.5);  // Thicker, darker edges
    mlp.visualize_network_with_config("network_custom_edges", "png", &edge_config).unwrap();
    println!("  [+] Saved to network_custom_edges.png\n");

    // Example 8: Professional theme (subtle colors)
    println!("Example 8: Professional Theme");
    let professional_config = NetworkVisualizationConfig::with_colors(
        "ghostwhite", "steelblue",        // Input: ghost white bg, steel blue neurons
        "floralwhite", "cadetblue",       // Hidden: floral white bg, cadet blue neurons
        "ivory", "darkseagreen",          // Output: ivory bg, dark sea green neurons
    ).with_edges("slategray", 0.8);
    mlp.visualize_network_with_config("network_professional", "png", &professional_config).unwrap();
    println!("  [+] Saved to network_professional.png\n");

    println!("=== Summary ===");
    println!("Generated 8 visualizations with different color schemes:");
    println!("  • network_default.png - Default blue/yellow/green");
    println!("  • network_purple.png - Purple/pink theme");
    println!("  • network_warm.png - Red/orange theme");
    println!("  • network_cool.png - Blue/green theme");
    println!("  • network_mono.png - Grayscale theme");
    println!("  • network_vibrant.png - High saturation colors");
    println!("  • network_custom_edges.png - Custom edge styling");
    println!("  • network_professional.png - Professional subtle colors");
    
    println!("\n Tip: Any CSS/X11 color names work!");
    println!("   See: https://graphviz.org/doc/info/colors.html");
}

