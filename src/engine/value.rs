//! Scalar value node that participates in automatic differentiation.

use std::cell::RefCell;
use std::rc::Rc;
use std::fmt::Display;
use std::ops::{Add, Sub, Mul, Div};

#[derive(Debug, Clone, Copy)]
pub enum OP {
    ADD,
    SUB,
    MUL,
    DIV,
    NONE,
}


impl Display for OP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Operation({})", match self {
            OP::ADD => "ADD",
            OP::SUB => "SUB",
            OP::MUL => "MUL",
            OP::DIV => "DIV",
            OP::NONE => "NONE",
        })
    }
}

//the actual reference everyone should work with
#[derive(Debug, Clone)]
pub struct Node {
    value: Rc<RefCell<Value>>
}

impl Node {
    pub fn new(value: f64) -> Self {
        Self::with_children(value, Vec::new(), OP::NONE)
    }
// to create a node with children
    fn with_children(value: f64, children: Vec<Node>, operation: OP) -> Self {
        Self { value: Rc::new(RefCell::new(Value::new(value, children, operation))) }
    }

    // to get the value of the node
    pub fn get_value(&self) -> f64 {
        self.value.borrow().get_value()
    }

    pub fn get_children(&self) -> Vec<Node> {
        self.value.borrow().children.clone()
    }

    pub fn get_gradient(&self) -> f64 {
        self.value.borrow().get_gradient()
    }

    pub fn set_gradient(&mut self, gradient: f64) {
        self.value.borrow_mut().set_gradient(gradient);
    }

    pub fn get_operation(&self) -> OP {
        self.value.borrow().get_operation()
    }
}


// Scalar value tracked by the autograd engine.

#[derive(Debug)]
pub struct Value {
    value: f64,
    gradient: f64,
    children: Vec<Node>,
    operation: OP,
}


impl Value {
    /// Construct a value node from raw data.
    pub fn new(value: f64, children: Vec<Node>, operation: OP) -> Self {
        Self { value, gradient: 0.0, children, operation }
    }

    pub fn with_children(value: f64, children: Vec<Node>, operation: OP) -> Self {
        Self { value, gradient: 0.0, children, operation }
    }


    pub fn get_value(&self) -> f64 {
        self.value
    }
    
    pub fn get_children(&self) -> Vec<Node> {
        self.children.clone()
    }

    pub fn get_gradient(&self) -> f64 {
        self.gradient
    }

    pub fn get_operation(&self) -> OP {
        self.operation
    }

    pub fn set_gradient(&mut self, gradient: f64) {
        self.gradient = gradient;
    }
}

impl From<f64> for Node {
    fn from(value: f64)->Self {
        Self::new(value)
    }

}

impl From<f32> for Node {
    fn from(value: f32) -> Self {
        Self::new(value as f64)
    }
}

impl From<i32> for Node {
    fn from(value: i32) -> Self {
        Self::new(value as f64)
    }
}

impl From<i64> for Node {
    fn from(value: i64) -> Self {
        Self::new(value as f64)
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(val={}, grad={}, children={}, operation={})", self.get_value(), self.get_gradient(), self.get_children().len(), self.get_operation())
    }
}

impl Add for Node {
    type Output = Node;

    fn add(self, other: Node) -> Node {
        Node::with_children(self.get_value() + other.get_value(), vec![self, other], OP::ADD)
    }
}

impl Sub for Node {
    type Output = Node;

    fn sub(self, other: Node) -> Node {
        Node::with_children(self.get_value() - other.get_value(), vec![self, other], OP::SUB)
    }
}

impl Mul for Node {
    type Output = Node;

    fn mul(self, other: Node) -> Node {
        Node::with_children(self.get_value() * other.get_value(), vec![self, other], OP::MUL)
    }
}

impl Div for Node {
    type Output = Node;

    fn div(self, other: Node) -> Node {
        Node::with_children(self.get_value() / other.get_value(), vec![self, other], OP::DIV)
    }
}