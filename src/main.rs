use rusty_axon::engine::{ComputationGraph, Node};

fn main() {
    println!("rusty-axon skeleton ready; start implementing micrograd logic.");

    // Sanity check that core structs are visible to the binary target.
    let _ = std::mem::size_of::<Node>();
    let _ = std::mem::size_of::<ComputationGraph>();
    let a = Node::from(2.0);
    let b = Node::from(-3);
    let c = a.clone() + b.clone();
    let d = c.pow(2.0);
    //let mut e = d.exp();
    let e = Node::from(1.0);
    let mut f = ((e.clone()*2.0).exp()-1.0)/((e.clone()*2.0).exp()+1.0);
    println!("a: {}", a);
    println!("b: {}", b);
    println!("c: {}", c);
    println!("d: {}", d);
    println!("e: {}", e);
    println!("f: {}", f);
    println!("After backward");
    f.backward();
    println!("a: {}", a);
    println!("b: {}", b);
    println!("c: {}", c);
    println!("d: {}", d);
    println!("e: {}", e);
    println!("f: {}", f);
}
