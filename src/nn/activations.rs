#[cfg(feature = "train")]
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
    /// Train-path application: builds the scalar `Node` chain that backprops
    /// through the activation.  Inference builds use
    /// [`Activations::apply_f32_inplace`] instead.
    #[cfg(feature = "train")]
    pub fn apply(&self, x: Node) -> Node {
        match self {
            Activations::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activations::ReLU => x.relu(),
            Activations::Tanh => {
                let x2 = x.clone() * 2.0;
                let exp_x = x2.exp();
                (exp_x.clone() - 1.0) / (exp_x + 1.0)
            }
            Activations::Swish => x.clone() / (1.0 + (-x.clone()).exp()),
            Activations::None => x,
        }
    }

    /// Pure-`f32` element-wise activation, applied in place.  Always
    /// available — inference builds use this exclusively.
    pub fn apply_f32_inplace(&self, x: &mut [f32]) {
        match self {
            Activations::None => {}
            Activations::ReLU => {
                for v in x.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            Activations::Sigmoid => {
                for v in x.iter_mut() {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
            }
            Activations::Tanh => {
                for v in x.iter_mut() {
                    *v = v.tanh();
                }
            }
            Activations::Swish => {
                for v in x.iter_mut() {
                    *v = *v / (1.0 + (-*v).exp());
                }
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
