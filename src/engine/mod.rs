//! Core autograd engine pieces.

pub mod ops;
pub mod value;

#[cfg(test)]
mod tests;

pub use value::{Node, Value};
