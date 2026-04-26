//! Layer-oriented neural network visualization.

use std::fs::File;
use std::io::Write;

/// Color scheme for a layer
#[derive(Clone)]
pub struct LayerColors {
    /// Background color of the layer cluster
    pub background: String,
    /// Fill color of the neurons
    pub neuron: String,
}

impl LayerColors {
    pub fn new(background: &str, neuron: &str) -> Self {
        Self {
            background: background.to_string(),
            neuron: neuron.to_string(),
        }
    }
}

/// Configuration for neural network visualization
pub struct NetworkVisualizationConfig {
    pub show_weights: bool,
    pub show_bias: bool,
    pub neuron_size: f64,
    pub layer_spacing: f64,
    pub neuron_spacing: f64,

    // Colors for different layer types
    pub input_colors: LayerColors,
    pub hidden_colors: LayerColors,
    pub output_colors: LayerColors,

    // Edge appearance
    pub edge_color: String,
    pub edge_width: f64,
}

impl Default for NetworkVisualizationConfig {
    fn default() -> Self {
        Self {
            show_weights: false,
            show_bias: false,
            neuron_size: 0.6,
            layer_spacing: 3.0,
            neuron_spacing: 1.0,

            // Default color scheme
            input_colors: LayerColors::new("aliceblue", "lightblue"),
            hidden_colors: LayerColors::new("lightyellow", "gold"),
            output_colors: LayerColors::new("honeydew", "lightgreen"),

            edge_color: "gray70".to_string(),
            edge_width: 0.5,
        }
    }
}

impl NetworkVisualizationConfig {
    /// Create a custom color scheme
    pub fn with_colors(
        input_bg: &str,
        input_neuron: &str,
        hidden_bg: &str,
        hidden_neuron: &str,
        output_bg: &str,
        output_neuron: &str,
    ) -> Self {
        Self {
            input_colors: LayerColors::new(input_bg, input_neuron),
            hidden_colors: LayerColors::new(hidden_bg, hidden_neuron),
            output_colors: LayerColors::new(output_bg, output_neuron),
            ..Default::default()
        }
    }

    /// Set edge appearance
    pub fn with_edges(mut self, color: &str, width: f64) -> Self {
        self.edge_color = color.to_string();
        self.edge_width = width;
        self
    }
}

/// Generate a layer-oriented DOT graph for a neural network
pub fn generate_network_dot(
    layer_sizes: &[usize],
    layer_names: &[String],
    activation_names: &[String],
    config: &NetworkVisualizationConfig,
) -> String {
    let mut dot = String::from("digraph NeuralNetwork {\n");
    dot.push_str("    rankdir=LR;\n");
    dot.push_str("    splines=line;\n");
    dot.push_str("    nodesep=0.8;\n");
    dot.push_str("    ranksep=2.5;\n");
    dot.push_str("    node [shape=circle, fixedsize=true, width=0.8, style=filled];\n");
    dot.push_str(&format!(
        "    edge [color={}, penwidth={}];\n\n",
        config.edge_color, config.edge_width
    ));

    // Create subgraphs for each layer
    for (layer_idx, &layer_size) in layer_sizes.iter().enumerate() {
        let layer_name = &layer_names[layer_idx];

        // Build label with activation function if available
        let label = if layer_idx < activation_names.len() && !activation_names[layer_idx].is_empty()
        {
            format!("{}\\n({})", layer_name, activation_names[layer_idx])
        } else {
            layer_name.clone()
        };

        dot.push_str(&format!("    subgraph cluster_{} {{\n", layer_idx));
        dot.push_str(&format!("        label=\"{}\";\n", label));
        dot.push_str("        style=rounded;\n");
        dot.push_str("        fontsize=14;\n");
        dot.push_str("        fontname=\"Arial Bold\";\n");

        // Determine colors based on layer type using config
        let layer_colors = if layer_idx == 0 {
            &config.input_colors // Input layer
        } else if layer_idx == layer_sizes.len() - 1 {
            &config.output_colors // Output layer
        } else {
            &config.hidden_colors // Hidden layers
        };

        dot.push_str(&format!("        bgcolor={};\n", layer_colors.background));
        dot.push_str(&format!(
            "        node [fillcolor={}];\n",
            layer_colors.neuron
        ));

        // Create nodes for this layer
        dot.push_str("        { rank=same; ");
        for neuron_idx in 0..layer_size {
            let node_id = format!("L{}N{}", layer_idx, neuron_idx);
            dot.push_str(&format!("{}; ", node_id));
        }
        dot.push_str("}\n");
        dot.push_str("    }\n\n");
    }

    // Create edges between layers
    for layer_idx in 0..layer_sizes.len() - 1 {
        let current_size = layer_sizes[layer_idx];
        let next_size = layer_sizes[layer_idx + 1];

        for current_neuron in 0..current_size {
            for next_neuron in 0..next_size {
                let from_id = format!("L{}N{}", layer_idx, current_neuron);
                let to_id = format!("L{}N{}", layer_idx + 1, next_neuron);
                dot.push_str(&format!("    {} -> {};\n", from_id, to_id));
            }
        }
        dot.push_str("\n");
    }

    dot.push_str("}\n");
    dot
}

/// Save the network visualization to a DOT file
pub fn save_network_graph(
    filename: &str,
    layer_sizes: &[usize],
    layer_names: &[String],
    activation_names: &[String],
    config: &NetworkVisualizationConfig,
) -> std::io::Result<()> {
    let dot = generate_network_dot(layer_sizes, layer_names, activation_names, config);
    let mut file = File::create(filename)?;
    file.write_all(dot.as_bytes())?;
    println!("[+] Network graph saved to {}", filename);
    println!("  Render with: dot -Tpng {} -o network.png", filename);
    Ok(())
}

/// Check if Graphviz is installed
pub fn check_graphviz() -> bool {
    std::process::Command::new("dot").arg("-V").output().is_ok()
}

/// Render the network graph to an image file
pub fn render_network_to(
    output_name: &str,
    format: &str,
    layer_sizes: &[usize],
    layer_names: &[String],
    activation_names: &[String],
    config: &NetworkVisualizationConfig,
) -> std::io::Result<()> {
    let dot_file = format!("{}.dot", output_name);
    let output_file = format!("{}.{}", output_name, format);

    // Save DOT file first
    save_network_graph(
        &dot_file,
        layer_sizes,
        layer_names,
        activation_names,
        config,
    )?;

    // Check if graphviz is available
    if !check_graphviz() {
        println!("[-] Graphviz not found!");
        println!("  Download from: https://graphviz.org/download/");
        println!("  Windows: winget install graphviz or choco install graphviz");
        println!("  Mac: brew install graphviz");
        println!("  Linux: sudo apt install graphviz");
        println!("\n  You can still view the .dot file at: http://www.webgraphviz.com/");
        return Ok(());
    }

    // Validate format
    let valid_formats = ["png", "svg", "pdf", "jpg", "jpeg"];
    if !valid_formats.contains(&format) {
        println!("[-] Unsupported format: {}", format);
        println!("  Supported formats: png, svg, pdf, jpg");
        return Ok(());
    }

    // Render with dot command
    let format_arg = format!("-T{}", format);
    let result = std::process::Command::new("dot")
        .args(&[&format_arg, &dot_file, "-o", &output_file])
        .output();

    match result {
        Ok(output) => {
            if output.status.success() {
                println!("[+] Network graph rendered to {}", output_file);

                // Show file size
                if let Ok(metadata) = std::fs::metadata(&output_file) {
                    let size_kb = metadata.len() / 1024;
                    println!("  Size: {} KB", size_kb);
                }
            } else {
                let error = String::from_utf8_lossy(&output.stderr);
                println!("[-] Rendering failed: {}", error);
            }
            Ok(())
        }
        Err(e) => {
            println!("[-] Could not render graph: {}", e);
            Ok(())
        }
    }
}
