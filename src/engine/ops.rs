//! Definitions of differentiable operations that can be applied to values.

use std::fmt::Display;
use crate::engine::value::Node;

/// Every differentiable operation should describe how to perform the forward
/// computation and how to propagate gradients backward.
#[derive(Debug, Clone)]
pub enum Operation {
    Add { left: Node, right: Node },
    Sub { minuend: Node, subtrahend: Node },
    Mul { left: Node, right: Node },
    Div { dividend: Node, divisor: Node },
    Pow { base: Node, exponent: f64 },
    Exp { exponent: Node },
    Neg { operand: Node },
    Log { base: f64, operand: Node },
    ReLU { input: Node },
    None,
}


impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Operation({})", match self {
            Operation::Add { .. } => "ADD",
            Operation::Sub { .. } => "SUB",
            Operation::Mul { .. } => "MUL",
            Operation::Div { .. } => "DIV",
            Operation::Pow { .. } => "POW",
            Operation::Exp { .. } => "EXP",
            Operation::Neg { .. } => "NEG",
            Operation::Log { .. } => "LOG",
            Operation::ReLU { .. } => "RELU",
            Operation::None => "NONE",
        })
    }
}
