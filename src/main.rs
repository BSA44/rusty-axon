use rusty_axon::engine::{ComputationGraph, Node};

fn main() {
    println!("rusty-axon skeleton ready; start implementing micrograd logic.");

    // Sanity check that core structs are visible to the binary target.
    let _ = std::mem::size_of::<Node>();
    let _ = std::mem::size_of::<ComputationGraph>();
    let a = Node::from(2.0);
    let b = Node::from(-3);
    let c = a.clone() + b.clone();
    println!("a: {}", a);
    println!("b: {}", b);
    println!("c: {}", c);
}
