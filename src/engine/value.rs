//! Scalar value node that participates in automatic differentiation.
//use std::collections::HashSet;
use std::cell::RefCell;
use std::rc::Rc;
use std::fmt::Display;
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::hash::{Hash, Hasher};
use std::collections::HashSet;
use crate::engine::ops::Operation;

macro_rules! impl_op_scalar {
    ($op_trait:ident, $op_method:ident, $scalar_ty:ty) => {
        // Node op Scalar (e.g., node * 2.0)
        impl $op_trait<$scalar_ty> for Node {
            type Output = Node;
            
            fn $op_method(self, rhs: $scalar_ty) -> Node {
                self.$op_method(Node::from(rhs))
            }
        }
        
        // Scalar op Node (e.g., 2.0 * node)
        impl $op_trait<Node> for $scalar_ty {
            type Output = Node;
            
            fn $op_method(self, rhs: Node) -> Node {
                Node::from(self).$op_method(rhs)
            }
        }
    };
}

macro_rules! impl_ops_for_scalar {
    ($scalar_ty:ty) => {
        impl_op_scalar!(Add, add, $scalar_ty);
        impl_op_scalar!(Sub, sub, $scalar_ty);
        impl_op_scalar!(Mul, mul, $scalar_ty);
        impl_op_scalar!(Div, div, $scalar_ty);
    };
}

//the actual reference everyone should work with
#[derive(Debug, Clone)]
pub struct Node {
    value: Rc<RefCell<Value>>
}

impl Node {
    pub fn new(value: f64) -> Self {
        Self::with_operation(value, Operation::None)
    }

    fn with_operation(value: f64, operation: Operation) -> Self {
        Self { value: Rc::new(RefCell::new(Value::new(value, operation))) }
    }

    
    // to get the value of the node
    pub fn get_value(&self) -> f64 {
        self.value.borrow().get_value()
    }


    pub fn get_gradient(&self) -> f64 {
        self.value.borrow().get_gradient()
    }

    pub fn set_gradient(&mut self, gradient: f64) {
        self.value.borrow_mut().set_gradient(gradient);
    }

    pub fn add_gradient(&self, gradient: f64) {
        self.value.borrow_mut().gradient += gradient;
    }

    pub fn zero_gradient(&self) {
        self.value.borrow_mut().set_gradient(0.0);
    }

    pub fn get_operation(&self) -> Operation {
        self.value.borrow().get_operation()
    }

    fn build_topo(&self ) -> Vec<Node> {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        self.build_topo_recursive(&mut topo, &mut visited);
        return topo;
    }
    
    fn build_topo_recursive(&self, topo:&mut Vec<Node>, visited:&mut HashSet<Node>) {
        if visited.contains(self) {
            return;
        }
        visited.insert(self.clone());
        
        let operation = self.get_operation();
        match operation {
            Operation::Add { left, right } 
            | Operation::Mul { left, right } =>
            {
                left.build_topo_recursive(topo, visited);
                right.build_topo_recursive(topo, visited);
            },
            Operation::Sub { minuend, subtrahend } => {
                minuend.build_topo_recursive(topo, visited);
                subtrahend.build_topo_recursive(topo, visited);
            },
            Operation::Div { dividend, divisor } => {
                dividend.build_topo_recursive(topo, visited);
                divisor.build_topo_recursive(topo, visited);
            },
            Operation::Pow { base, exponent } => {
                base.build_topo_recursive(topo, visited);
            },
            Operation::Exp { exponent } => {
                exponent.build_topo_recursive(topo, visited);
            },
            Operation::Neg { operand } => {
                operand.build_topo_recursive(topo, visited);
            },
            Operation::None => {
            }
        }
        topo.push(self.clone());
    }


    pub fn backward(&mut self) {
        self.set_gradient(1.0);
        let topo = self.build_topo();
        for node in topo.iter().rev() {
            let node_borrow = node.value.borrow();
            //out gradient
            let grad = node_borrow.get_gradient();
            
            match &node_borrow.get_operation() {
                Operation::Add { left, right } =>
                {
                    drop(node_borrow);
                    left.add_gradient(grad);
                    right.add_gradient(grad);
                },
                Operation::Div { dividend, divisor } =>
                {
                    drop(node_borrow);
                    dividend.add_gradient(grad*(1.0/divisor.get_value()));
                    divisor.add_gradient(-(grad)*(dividend.get_value()/(divisor.get_value()*divisor.get_value())));
                },
                Operation::Mul { left, right } =>
                {
                    drop(node_borrow);
                    left.add_gradient(grad*right.get_value());
                    right.add_gradient(grad*left.get_value());
                },
                Operation::Sub { minuend, subtrahend } =>
                {
                    drop(node_borrow);
                    minuend.add_gradient(grad);
                    subtrahend.add_gradient(-grad);
                },
                Operation::Pow { base, exponent } =>
                {
                    drop(node_borrow);
                    base.add_gradient(grad*exponent*base.get_value().powf(exponent-1.0));
                },
                Operation::Exp { exponent } =>
                {
                    let exp_result = node_borrow.get_value();
                    drop(node_borrow);
                    exponent.add_gradient(grad*exp_result);
                },
                Operation::Neg { operand } =>
                {
                    drop(node_borrow);
                    operand.add_gradient(-grad);
                },
                Operation::None =>
                {
                    drop(node_borrow);
                }
            }
        }
    }


    pub fn pow(&self, exponent: f64) -> Node {
        Node::with_operation(self.get_value().powf(exponent), Operation::Pow { base: self.clone(), exponent })
    }

    pub fn exp(&self) -> Node {
        Node::with_operation(self.get_value().exp(), Operation::Exp { exponent: self.clone() })
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.value, &other.value)
    }
}

impl Eq for Node {}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        //hash based on the address of the value
        Rc::as_ptr(&self.value).hash(state);
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(val={}, grad={}, operation={})", self.get_value(), self.get_gradient(), self.get_operation())
    }
}


impl Add for Node {
    type Output = Node;

    fn add(self, other: Node) -> Node {
        let new_val = self.get_value() + other.get_value();
        Node::with_operation(new_val, Operation::Add { left: self, right: other })
    }
}

impl Sub for Node {
    type Output = Node;

    fn sub(self, other: Node) -> Node {
        let new_val = self.get_value() - other.get_value();
        Node::with_operation(new_val,  Operation::Sub { minuend: self, subtrahend: other })
    }
}

impl Mul for Node {
    type Output = Node;

    fn mul(self, other: Node) -> Node {
        let new_val = self.get_value() * other.get_value();
        Node::with_operation(new_val, Operation::Mul { left: self, right: other })
    }   
}

impl Div for Node {
    type Output = Node;

    fn div(self, other: Node) -> Node {
        let new_val = self.get_value() / other.get_value();
        Node::with_operation(new_val, Operation::Div { dividend: self, divisor: other })
    }
}

impl Neg for Node {
    type Output = Node;

    fn neg(self) -> Node {
        Node::with_operation(-self.get_value(), Operation::Neg { operand: self })
    }
}

// Scalar value tracked by the autograd engine.

#[derive(Debug)]
pub struct Value {
    value: f64,
    gradient: f64,
    operation: Operation,
}


impl Value {
    /// Construct a value node from raw data.
    pub fn new(value: f64, operation: Operation) -> Self {
        Self { value, gradient: 0.0, operation }
    }

    
    pub fn with_operation(value: f64, operation: Operation) -> Self {
        Self { value, gradient: 0.0, operation }
    }


    pub fn get_value(&self) -> f64 {
        self.value
    }
    
    pub fn get_gradient(&self) -> f64 {
        self.gradient
    }

    pub fn get_operation(&self) -> Operation {
        self.operation.clone()
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

// Invoke macros to generate scalar operation implementations
//impl_ops_for_scalar!(i64);
//impl_ops_for_scalar!(f32);
//impl_ops_for_scalar!(i32);
impl_ops_for_scalar!(f64);

