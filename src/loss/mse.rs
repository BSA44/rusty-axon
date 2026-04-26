use crate::engine::value::Node;
use crate::loss::loss::Loss;

pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn forward(&self, predictions: &[Node], targets: &[Node]) -> Node {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have the same length"
        );
        let mut loss = Node::new(0.0);
        for (prediction, target) in predictions.iter().zip(targets.iter()) {
            loss = loss + (prediction.clone() - target.clone()).pow(2.0);
        }
        loss / predictions.len() as f64
    }
}
