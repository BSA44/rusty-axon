use rusty_axon::engine::{ComputationGraph, Value};

fn main() {
    println!("rusty-axon skeleton ready; start implementing micrograd logic.");

    // Sanity check that core structs are visible to the binary target.
    let _ = std::mem::size_of::<Value>();
    let _ = std::mem::size_of::<ComputationGraph>();
}
