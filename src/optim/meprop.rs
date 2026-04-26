use crate::engine::value::Node;
use crate::optim::optimizer::Optimizer;

pub struct MeProp {
    learning_rate: f64,
    parameters: Vec<Node>,
    top_k: f32,
}

impl MeProp {
    pub fn new(learning_rate: f64, parameters: Vec<Node>, top_k: f32) -> Self {
        Self {
            learning_rate,
            parameters,
            top_k: top_k.clamp(0.0, 1.0),
        }
    }
}

impl Optimizer for MeProp {
    fn step(&mut self) {
        let total_params = self.parameters.len();
        if total_params == 0 {
            return;
        }

        let k = (self.top_k * (total_params as f32)).ceil() as usize;
        let k = k.max(1).min(total_params); // Also cap at total_params

        let mut param_grads: Vec<(&mut Node, f64)> = self
            .parameters
            .iter_mut()
            .map(|p| {
                let grad = p.get_gradient().abs();
                (p, grad)
            })
            .collect();

        param_grads.sort_by(|a, b| b.1.total_cmp(&a.1));

        for (p, _) in param_grads.into_iter().take(k) {
            p.set_value(p.get_value() - self.learning_rate * p.get_gradient());
        }
    }

    fn zero_state(&mut self) {
        for param in self.parameters.iter_mut() {
            param.zero_gradient();
        }
    }
}
