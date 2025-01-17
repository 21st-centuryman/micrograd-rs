use micrograd::engine::Value;
use micrograd::graph::draw_dot;
use micrograd::nn::Neuron;

fn main() {
    let n = Neuron::new(2, true);
    let x = vec![Value::from(1.0), Value::from(-2.0)];
    let y = n.forward(&x);
    let _ = &y.backward();
    draw_dot(y.clone());
}
