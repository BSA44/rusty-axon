
use crate::engine::Node;
#[derive(Debug, Clone)]

pub enum Activations {
    Sigmoid,
    //ReLU, add them later
    Tanh,
    //LeakyReLU(f64),
    //ELU(f64),
    Swish,
    None,
}

impl Activations {
    pub fn apply(&self, x: Node) -> Node {
        match self {
            Activations::Sigmoid => 
            {
             1.0 / (1.0 + (-x).exp())
            },
            Activations::Tanh =>
            {
                let x2 = x.clone() * 2.0;
                let exp_x = x2.exp();
                (exp_x.clone() - 1.0) / (exp_x + 1.0)
            },
            Activations::Swish =>
            {
                x.clone() / (1.0 + (-x.clone()).exp())
            },
            Activations::None => {
                x
            }
        }
    }
}