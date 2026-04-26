//! Scalar value node that participates in automatic differentiation.
//use std::collections::HashSet;
use crate::engine::ops::Operation;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Display;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

macro_rules! impl_op_scalar {
    ($op_trait:ident, $op_method:ident, $scalar_ty:ty) => {
        // Node op Scalar (e.g., node * 2.0)
        impl $op_trait<$scalar_ty> for Node {
            type Output = Node;

            fn $op_method(self, rhs: $scalar_ty) -> Node {
                self.$op_method(Node::from(rhs))
            }
        }

        // Scalar op Node (e.g., 2.0 * node)
        impl $op_trait<Node> for $scalar_ty {
            type Output = Node;

            fn $op_method(self, rhs: Node) -> Node {
                Node::from(self).$op_method(rhs)
            }
        }
    };
}

macro_rules! impl_ops_for_scalar {
    ($scalar_ty:ty) => {
        impl_op_scalar!(Add, add, $scalar_ty);
        impl_op_scalar!(Sub, sub, $scalar_ty);
        impl_op_scalar!(Mul, mul, $scalar_ty);
        impl_op_scalar!(Div, div, $scalar_ty);
    };
}

//the actual reference everyone should work with
#[derive(Debug, Clone)]
pub struct Node {
    value: Rc<RefCell<Value>>,
}

impl Node {
    pub fn new(value: f32) -> Self {
        Self::with_operation(value, Operation::None)
    }

    fn with_operation(value: f32, operation: Operation) -> Self {
        Self {
            value: Rc::new(RefCell::new(Value::new(value, operation))),
        }
    }

    pub fn set_value(&mut self, value: f32) {
        self.value.borrow_mut().set_value(value);
    }

    // to get the value of the node
    pub fn get_value(&self) -> f32 {
        self.value.borrow().get_value()
    }

    pub fn get_gradient(&self) -> f32 {
        self.value.borrow().get_gradient()
    }

    pub fn set_gradient(&mut self, gradient: f32) {
        self.value.borrow_mut().set_gradient(gradient);
    }

    pub fn add_gradient(&self, gradient: f32) {
        self.value.borrow_mut().gradient += gradient;
    }

    pub fn zero_gradient(&self) {
        self.value.borrow_mut().set_gradient(0.0);
    }

    pub fn get_operation(&self) -> Operation {
        self.value.borrow().get_operation()
    }

    fn build_topo(&self) -> Vec<Node> {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        self.build_topo_recursive(&mut topo, &mut visited);
        return topo;
    }

    fn build_topo_recursive(&self, topo: &mut Vec<Node>, visited: &mut HashSet<Node>) {
        if visited.contains(self) {
            return;
        }
        visited.insert(self.clone());

        let operation = self.get_operation();
        match operation {
            Operation::Add { left, right } | Operation::Mul { left, right } => {
                left.build_topo_recursive(topo, visited);
                right.build_topo_recursive(topo, visited);
            }
            Operation::Sub {
                minuend,
                subtrahend,
            } => {
                minuend.build_topo_recursive(topo, visited);
                subtrahend.build_topo_recursive(topo, visited);
            }
            Operation::Div { dividend, divisor } => {
                dividend.build_topo_recursive(topo, visited);
                divisor.build_topo_recursive(topo, visited);
            }
            Operation::Pow { base, exponent: _ } => {
                base.build_topo_recursive(topo, visited);
            }
            Operation::Exp { exponent } => {
                exponent.build_topo_recursive(topo, visited);
            }
            Operation::Neg { operand } => {
                operand.build_topo_recursive(topo, visited);
            }
            Operation::Log { base: _, operand } => {
                operand.build_topo_recursive(topo, visited);
            }
            Operation::ReLU { input } => {
                input.build_topo_recursive(topo, visited);
            }
            Operation::None => {}
        }
        topo.push(self.clone());
    }

    pub fn backward(&mut self) {
        self.set_gradient(1.0);
        let topo = self.build_topo();
        for node in topo.iter().rev() {
            let node_borrow = node.value.borrow();
            //out gradient
            let grad = node_borrow.get_gradient();

            match &node_borrow.get_operation() {
                Operation::Add { left, right } => {
                    drop(node_borrow);
                    left.add_gradient(grad);
                    right.add_gradient(grad);
                }
                Operation::Div { dividend, divisor } => {
                    drop(node_borrow);
                    dividend.add_gradient(grad * (1.0 / divisor.get_value()));
                    divisor.add_gradient(
                        -(grad)
                            * (dividend.get_value() / (divisor.get_value() * divisor.get_value())),
                    );
                }
                Operation::Mul { left, right } => {
                    drop(node_borrow);
                    left.add_gradient(grad * right.get_value());
                    right.add_gradient(grad * left.get_value());
                }
                Operation::Sub {
                    minuend,
                    subtrahend,
                } => {
                    drop(node_borrow);
                    minuend.add_gradient(grad);
                    subtrahend.add_gradient(-grad);
                }
                Operation::Pow { base, exponent } => {
                    drop(node_borrow);
                    base.add_gradient(grad * exponent * base.get_value().powf(exponent - 1.0));
                }
                Operation::Exp { exponent } => {
                    let exp_result = node_borrow.get_value();
                    drop(node_borrow);
                    exponent.add_gradient(grad * exp_result);
                }
                Operation::Neg { operand } => {
                    drop(node_borrow);
                    operand.add_gradient(-grad);
                }
                Operation::Log { base, operand } => {
                    drop(node_borrow);
                    operand.add_gradient(grad / (operand.get_value() * base.ln()));
                }
                Operation::ReLU { input } => {
                    drop(node_borrow);
                    // ReLU gradient: 1 if input > 0, else 0
                    if input.get_value() > 0.0 {
                        input.add_gradient(grad);
                    }
                    // else gradient is 0, so we don't add anything
                }
                Operation::None => {
                    drop(node_borrow);
                }
            }
        }
    }

    pub fn pow(&self, exponent: f32) -> Node {
        Node::with_operation(
            self.get_value().powf(exponent),
            Operation::Pow {
                base: self.clone(),
                exponent,
            },
        )
    }

    pub fn exp(&self) -> Node {
        Node::with_operation(
            self.get_value().exp(),
            Operation::Exp {
                exponent: self.clone(),
            },
        )
    }

    pub fn log(&self, base: f32) -> Node {
        Node::with_operation(
            self.get_value().log(base),
            Operation::Log {
                base,
                operand: self.clone(),
            },
        )
    }

    pub fn relu(&self) -> Node {
        let value = self.get_value();
        Node::with_operation(
            value.max(0.0),
            Operation::ReLU {
                input: self.clone(),
            },
        )
    }

    /// Get unique identifier for this node based on its memory address
    fn node_id(&self) -> String {
        format!("n{:x}", Rc::as_ptr(&self.value) as usize)
    }

    /// Generate a DOT graph visualization of the computation graph
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph G {\n");
        dot.push_str("    rankdir=LR;\n");
        dot.push_str("    node [style=filled];\n");
        dot.push_str("    edge [color=gray];\n\n");

        let mut visited = HashSet::new();
        self.build_dot_recursive(&mut dot, &mut visited);

        dot.push_str("}\n");
        dot
    }

    /// Recursively build DOT graph representation
    fn build_dot_recursive(&self, dot: &mut String, visited: &mut HashSet<String>) {
        let id = self.node_id();

        if visited.contains(&id) {
            return;
        }
        visited.insert(id.clone());

        // Create node label with value and gradient
        let label = format!(
            "val={:.4}\\ngrad={:.4}",
            self.get_value(),
            self.get_gradient()
        );

        // Determine color based on gradient magnitude
        let grad_abs = self.get_gradient().abs();
        let fillcolor = if grad_abs > 1.0 {
            "lightcoral"
        } else if grad_abs > 0.1 {
            "lightyellow"
        } else if grad_abs > 1e-10 {
            "lightblue"
        } else {
            "lightgray"
        };

        // Add this value node to the graph
        dot.push_str(&format!(
            "    {} [label=\"{}\" shape=box fillcolor={}];\n",
            id, label, fillcolor
        ));

        // Handle operations
        let operation = self.get_operation();
        match operation {
            Operation::Add { left, right } => {
                let op_id = format!("{}_add", id);
                dot.push_str(&format!(
                    "    {} [label=\"+\" shape=circle fillcolor=orange];\n",
                    op_id
                ));

                // Recurse on children
                left.build_dot_recursive(dot, visited);
                right.build_dot_recursive(dot, visited);

                // Add edges
                dot.push_str(&format!("    {} -> {};\n", left.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", right.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", op_id, id));
            }
            Operation::Sub {
                minuend,
                subtrahend,
            } => {
                let op_id = format!("{}_sub", id);
                dot.push_str(&format!(
                    "    {} [label=\"-\" shape=circle fillcolor=orange];\n",
                    op_id
                ));

                minuend.build_dot_recursive(dot, visited);
                subtrahend.build_dot_recursive(dot, visited);

                dot.push_str(&format!("    {} -> {};\n", minuend.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", subtrahend.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", op_id, id));
            }
            Operation::Mul { left, right } => {
                let op_id = format!("{}_mul", id);
                dot.push_str(&format!(
                    "    {} [label=\"×\" shape=circle fillcolor=lightgreen];\n",
                    op_id
                ));

                left.build_dot_recursive(dot, visited);
                right.build_dot_recursive(dot, visited);

                dot.push_str(&format!("    {} -> {};\n", left.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", right.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", op_id, id));
            }
            Operation::Div { dividend, divisor } => {
                let op_id = format!("{}_div", id);
                dot.push_str(&format!(
                    "    {} [label=\"÷\" shape=circle fillcolor=lightgreen];\n",
                    op_id
                ));

                dividend.build_dot_recursive(dot, visited);
                divisor.build_dot_recursive(dot, visited);

                dot.push_str(&format!("    {} -> {};\n", dividend.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", divisor.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", op_id, id));
            }
            Operation::Pow { base, exponent } => {
                let op_id = format!("{}_pow", id);
                dot.push_str(&format!(
                    "    {} [label=\"^{:.2}\" shape=circle fillcolor=plum];\n",
                    op_id, exponent
                ));

                base.build_dot_recursive(dot, visited);

                dot.push_str(&format!("    {} -> {};\n", base.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", op_id, id));
            }
            Operation::Exp { exponent } => {
                let op_id = format!("{}_exp", id);
                dot.push_str(&format!(
                    "    {} [label=\"exp\" shape=circle fillcolor=plum];\n",
                    op_id
                ));

                exponent.build_dot_recursive(dot, visited);

                dot.push_str(&format!("    {} -> {};\n", exponent.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", op_id, id));
            }
            Operation::Log { base, operand } => {
                let op_id = format!("{}_log", id);
                dot.push_str(&format!(
                    "    {} [label=\"log_{{{}}}\" shape=circle fillcolor=plum];\n",
                    op_id, base
                ));

                operand.build_dot_recursive(dot, visited);

                dot.push_str(&format!("    {} -> {};\n", operand.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", op_id, id));
            }
            Operation::Neg { operand } => {
                let op_id = format!("{}_neg", id);
                dot.push_str(&format!(
                    "    {} [label=\"-\" shape=circle fillcolor=orange];\n",
                    op_id
                ));

                operand.build_dot_recursive(dot, visited);

                dot.push_str(&format!("    {} -> {};\n", operand.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", op_id, id));
            }
            Operation::ReLU { input } => {
                let op_id = format!("{}_relu", id);
                dot.push_str(&format!(
                    "    {} [label=\"ReLU\" shape=circle fillcolor=lightsalmon];\n",
                    op_id
                ));

                input.build_dot_recursive(dot, visited);

                dot.push_str(&format!("    {} -> {};\n", input.node_id(), op_id));
                dot.push_str(&format!("    {} -> {};\n", op_id, id));
            }
            Operation::None => {
                // Leaf node - already added above
            }
        }
    }

    /// Save the computation graph to a DOT file
    pub fn save_graph(&self, filename: &str) -> std::io::Result<()> {
        let dot = self.to_dot();
        let mut file = File::create(filename)?;
        file.write_all(dot.as_bytes())?;
        println!("[+]  Graph saved to {}", filename);
        println!("  Render with: dot -Tpng {} -o graph.png", filename);
        println!("  Or view online: http://www.webgraphviz.com/");
        Ok(())
    }

    /// Check if Graphviz is installed
    pub fn check_graphviz() -> bool {
        std::process::Command::new("dot").arg("-V").output().is_ok()
    }

    /// Render the computation graph to an image file
    ///
    /// # Arguments
    /// * `output_name` - Base name for output files (without extension)
    /// * `format` - Output format: "png", "svg", "pdf", "jpg"
    ///
    /// # Example
    /// ```ignore
    /// let x = Node::from(2.0);
    /// let mut y = x.pow(2.0);
    /// y.backward();
    /// y.render_to("graph", "png").unwrap();  // Creates graph.png
    /// y.render_to("graph", "svg").unwrap();  // Creates graph.svg
    /// ```
    pub fn render_to(&self, output_name: &str, format: &str) -> std::io::Result<()> {
        let dot_file = format!("{}.dot", output_name);
        let output_file = format!("{}.{}", output_name, format);

        // Save DOT file first
        self.save_graph(&dot_file)?;

        // Check if graphviz is available
        if !Self::check_graphviz() {
            println!("[-] Graphviz not found!");
            println!("  Download from: https://graphviz.org/download/");
            println!("  Windows: winget install graphviz or choco install graphviz");
            println!("  Mac: brew install graphviz");
            println!("  Linux: sudo apt install graphviz");
            println!("\n  You can still view the .dot file at: http://www.webgraphviz.com/");
            return Ok(());
        }

        // Validate format
        let valid_formats = ["png", "svg", "pdf", "jpg", "jpeg", "gif"];
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
                    println!("[+] Graph rendered to {}", output_file);

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

    /// Render to PNG (convenience method)
    pub fn render_png(&self, output_name: &str) -> std::io::Result<()> {
        self.render_to(output_name, "png")
    }

    /// Render to SVG (convenience method)
    pub fn render_svg(&self, output_name: &str) -> std::io::Result<()> {
        self.render_to(output_name, "svg")
    }

    /// Render to PDF (convenience method)
    pub fn render_pdf(&self, output_name: &str) -> std::io::Result<()> {
        self.render_to(output_name, "pdf")
    }

    /// Legacy method - now calls render_png
    #[deprecated(since = "0.1.0", note = "Use render_png() or render_to() instead")]
    pub fn render_graph(&self, output_name: &str) -> std::io::Result<()> {
        self.render_png(output_name)
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.value, &other.value)
    }
}

impl Eq for Node {}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        //hash based on the address of the value
        Rc::as_ptr(&self.value).hash(state);
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value(val={}, grad={}, operation={})",
            self.get_value(),
            self.get_gradient(),
            self.get_operation()
        )
    }
}

impl Add for Node {
    type Output = Node;

    fn add(self, other: Node) -> Node {
        let new_val = self.get_value() + other.get_value();
        Node::with_operation(
            new_val,
            Operation::Add {
                left: self,
                right: other,
            },
        )
    }
}

impl Sub for Node {
    type Output = Node;

    fn sub(self, other: Node) -> Node {
        let new_val = self.get_value() - other.get_value();
        Node::with_operation(
            new_val,
            Operation::Sub {
                minuend: self,
                subtrahend: other,
            },
        )
    }
}

impl Mul for Node {
    type Output = Node;

    fn mul(self, other: Node) -> Node {
        let new_val = self.get_value() * other.get_value();
        Node::with_operation(
            new_val,
            Operation::Mul {
                left: self,
                right: other,
            },
        )
    }
}

impl Div for Node {
    type Output = Node;

    fn div(self, other: Node) -> Node {
        let new_val = self.get_value() / other.get_value();
        Node::with_operation(
            new_val,
            Operation::Div {
                dividend: self,
                divisor: other,
            },
        )
    }
}

impl Neg for Node {
    type Output = Node;

    fn neg(self) -> Node {
        Node::with_operation(-self.get_value(), Operation::Neg { operand: self })
    }
}

// Scalar value tracked by the autograd engine.

#[derive(Debug)]
pub struct Value {
    value: f32,
    gradient: f32,
    operation: Operation,
}

impl Value {
    /// Construct a value node from raw data.
    pub fn new(value: f32, operation: Operation) -> Self {
        Self {
            value,
            gradient: 0.0,
            operation,
        }
    }

    pub fn with_operation(value: f32, operation: Operation) -> Self {
        Self {
            value,
            gradient: 0.0,
            operation,
        }
    }

    pub fn set_value(&mut self, value: f32) {
        self.value = value;
    }

    pub fn get_value(&self) -> f32 {
        self.value
    }

    pub fn get_gradient(&self) -> f32 {
        self.gradient
    }

    pub fn get_operation(&self) -> Operation {
        self.operation.clone()
    }

    pub fn set_gradient(&mut self, gradient: f32) {
        self.gradient = gradient;
    }
}

impl From<f32> for Node {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

// Lossy convenience: callers passing f64 literals (`Node::from(2.0)`) keep
// working through Phase 0.5; the engine itself is f32 end-to-end.
impl From<f64> for Node {
    fn from(value: f64) -> Self {
        Self::new(value as f32)
    }
}

impl From<i32> for Node {
    fn from(value: i32) -> Self {
        Self::new(value as f32)
    }
}

impl From<i64> for Node {
    fn from(value: i64) -> Self {
        Self::new(value as f32)
    }
}

// Scalar op-with-Node impls only on `f32` — the engine's native precision
// after Phase 0.5. Adding an additional `f64` macro would make untyped float
// literals (e.g. `node * 2.0`) ambiguous between two equally-valid impls.
// Untyped literals constrain to `f32` here through the only available impl.
impl_ops_for_scalar!(f32);
