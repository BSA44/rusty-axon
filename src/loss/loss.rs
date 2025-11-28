use crate::engine::value::Node;

pub trait Loss {
    fn forward(&self, predictions: &[Node], targets: &[Node]) -> Node;
}
