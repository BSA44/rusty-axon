//! Definitions of differentiable operations that can be applied to values.

use crate::engine::value::Node;
use crate::nn::matmul::MatMulTape;
use std::fmt::Display;
use std::rc::Rc;

/// Every differentiable operation should describe how to perform the forward
/// computation and how to propagate gradients backward.
///
/// The scalar variants (`Add` ... `ReLU`) carry their operand `Node`s inline
/// — each is one fat pointer of payload — so the `Value` graph stays the
/// pure micrograd-style primary structure.  `MatMul` is the only fused
/// variant: every output `Node` of one matmul carries `(Rc<MatMulTape>, usize)`
/// and the actual `out_dim × in_dim` weight buffer lives exactly once inside
/// the shared tape.  See [`crate::nn::matmul::MatMulTape`] for the
/// dispatch protocol.
#[derive(Debug, Clone)]
pub enum Operation {
    Add { left: Node, right: Node },
    Sub { minuend: Node, subtrahend: Node },
    Mul { left: Node, right: Node },
    Div { dividend: Node, divisor: Node },
    Pow { base: Node, exponent: f32 },
    Exp { exponent: Node },
    Neg { operand: Node },
    Log { base: f32, operand: Node },
    ReLU { input: Node },
    /// One output of a fused matmul.  All outputs of the same matmul share
    /// the `Rc<MatMulTape>`; `output_index` selects which row of `W @ x + b`
    /// this `Node` represents.
    MatMul {
        tape: Rc<MatMulTape>,
        output_index: usize,
    },
    None,
}

impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Operation({})",
            match self {
                Operation::Add { .. } => "ADD",
                Operation::Sub { .. } => "SUB",
                Operation::Mul { .. } => "MUL",
                Operation::Div { .. } => "DIV",
                Operation::Pow { .. } => "POW",
                Operation::Exp { .. } => "EXP",
                Operation::Neg { .. } => "NEG",
                Operation::Log { .. } => "LOG",
                Operation::ReLU { .. } => "RELU",
                Operation::MatMul { .. } => "MATMUL",
                Operation::None => "NONE",
            }
        )
    }
}
