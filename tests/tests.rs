/*
----------------------------------------------------------------------------------
This is a test to make sure that the tests in the examples pass.
----------------------------------------------------------------------------------
*/

use micrograd::engine::{Activations, Value};
use micrograd::nn::{Layer, mlp};

#[test]
pub fn test_usage() {
    let a = Value::from(-4.0);
    let b = Value::from(2.0);

    let mut c = &a + &b;
    let mut d = &a * &b + b.pow(&Value::from(3.0));

    c = &Value::from(2.0) * &c + Value::from(1.0);
    c = Value::from(1.0) + &Value::from(2.0) * &c + (-&a);
    d = &d + &(&d * &Value::from(2.0)) + (&b + &a).relu();
    d = &d + &(&Value::from(3.0) * &d) + (&b - &a).relu();

    let e = &c - &d;
    let f = e.pow(&Value::from(2.0));
    let mut g = &f / &Value::from(2.0);
    g = g + &Value::from(10.0) / &f;

    assert_eq!(format!("{:.4}", g.data.borrow()), "24.7041");
    g.backward();
    assert_eq!(format!("{:.4}", a.grad.borrow()), "138.8338");
    assert_eq!(format!("{:.4}", b.grad.borrow()), "645.5773");
}

#[test]
fn train() {
    // Variables
    mlp!(4);
    let range = 2000;
    let adjust = -0.01;
    let ys = vec![1.0, -1.0, -1.0, 1.0]; // desired targets

    let n: MLP<3, 4, 4, 1> = MLP::new(Activations::Relu, Activations::Relu, Activations::Linear);

    let xs = vec![vec![2.0, 3.0, -1.0], vec![3.0, -1.0, 0.5], vec![0.5, 1.0, 1.0], vec![1.0, 1.0, -1.0]];

    for k in 0..range {
        // Forward pass
        let ypred: Vec<Value> = xs
            .iter()
            .map(|x| {
                let input_array: [Value; 3] = [Value::from(x[0]), Value::from(x[1]), Value::from(x[2])];
                n.forward(&input_array)[0].clone()
            })
            .collect();
        let loss: Value = ypred
            .clone()
            .into_iter()
            .zip(ys.iter().map(|y| Value::from(*y)))
            .map(|(yout, ygt)| (yout - ygt).pow(&2.0.into()))
            .sum();

        // Backward pass
        for p in n.parameters() {
            p.zero_grad();
        }
        loss.backward();

        // Update
        for p in n.parameters() {
            p.adjust(adjust);
        }

        if k == range {
            assert_eq!(format!("{:.3}", ypred[0].data()), "1.000");
            assert_eq!(format!("{:.3}", ypred[1].data()), "-1.000");
            assert_eq!(format!("{:.3}", ypred[2].data()), "-1.000");
            assert_eq!(format!("{:.3}", ypred[3].data()), "1.000");
        }
    }
}
