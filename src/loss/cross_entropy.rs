use crate::engine::value::Node;
use crate::loss::loss::Loss;
pub struct CrossEntropy {
    label_smoothing: f32,
}

impl CrossEntropy {
    pub fn new(label_smoothing: f32) -> Self {
        Self {
            label_smoothing: label_smoothing.clamp(0.0, 1.0),
        }
    }

    fn softmax(&self, logits: &[Node]) -> Vec<Node> {
        //finding max
        let max = logits
            .iter()
            .map(|l| l.get_value())
            .fold(f32::NEG_INFINITY, f32::max);

        //compute exp(logits-max) as Nodes
        //let max_node = Node::from(max);
        let exp_logits: Vec<Node> = logits.iter().map(|l| (l.clone() - max).exp()).collect();

        let sum_exps = exp_logits.iter().cloned().reduce(|a, b| a + b).unwrap();

        exp_logits
            .iter()
            .map(|e| e.clone() / sum_exps.clone())
            .collect() //collect will derive return type directly from function return
    }

    fn smooth_targets(&self, targets: &[Node]) -> Vec<Node> {
        let num_of_classes = targets.len();
        targets
            .iter()
            .map(|t| {
                (1.0 - self.label_smoothing) * t.clone()
                    + self.label_smoothing / (num_of_classes as f32)
            })
            .collect()
    }
}

impl Loss for CrossEntropy {
    fn forward(&self, logits: &[Node], targets: &[Node]) -> Node {
        assert_eq!(
            logits.len(),
            targets.len(),
            "Predictions and targets must have the same length"
        );
        let mut loss = Node::from(0.0);
        let probabilities = self.softmax(logits);
        let smooth_targets = self.smooth_targets(targets);

        for (probability, target) in probabilities.iter().zip(smooth_targets.iter()) {
            // Add a small epsilon before log to avoid log(0); 1e-7 is the
            // standard f32-safe clamp (matches PyTorch's CE epsilon).
            loss = loss
                + (-target.clone() * (probability.clone() + 1e-7).log(std::f32::consts::E));
        }
        loss / logits.len() as f32
    }
}
