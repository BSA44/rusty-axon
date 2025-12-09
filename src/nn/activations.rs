
use crate::engine::Node;
use std::fmt;

#[derive(Debug, Clone)]
pub enum Activations {
    Sigmoid,
    ReLU,
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
            Activations::ReLU =>
            {
                x.relu()
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

impl fmt::Display for Activations {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Activations::Sigmoid => write!(f, "Sigmoid"),
            Activations::ReLU => write!(f, "ReLU"),
            Activations::Tanh => write!(f, "Tanh"),
            Activations::Swish => write!(f, "Swish"),
            Activations::None => write!(f, "Linear"),
        }
    }
}