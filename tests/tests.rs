/*
----------------------------------------------------------------------------------
This is a test to make sure that the tests in the examples pass.
----------------------------------------------------------------------------------
*/

use micrograd::engine::{Activations, Value};
use micrograd::nn::{mlp, Layer};

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
#[test]
fn make_moons() {
    mlp!(4);
    let (x, y): (Vec<[f32; 2]>, Vec<f32>) = csv::ReaderBuilder::new()
        .from_path("./datasets/make_moons/make_moons.csv")
        .unwrap()
        .records()
        .map(|r| {
            let record = r.unwrap();
            let x_val = [
                record.get(0).unwrap().parse::<f32>().unwrap(),
                record.get(1).unwrap().parse::<f32>().unwrap(),
            ];
            let y_val = record.get(2).unwrap().parse::<f32>().unwrap();
            (x_val, y_val)
        })
        .unzip(); // Splits into two vectors

    let model: MLP<2, 16, 16, 1> = MLP::new(Activations::Relu, Activations::Relu, Activations::Linear);
    //let model = mlp!(2, 16, 16, 1);

    fn loss(xs: &[[f32; 2]], ys: &[f32], model: &MLP<2, 16, 16, 1>) -> Value {
        let inputs: Vec<Vec<Value>> = xs.iter().map(|xrow| vec![Value::from(xrow[0]), Value::from(xrow[1])]).collect();

        // forward the model to get scores
        let scores: Vec<Value> = inputs
            .iter()
            .map(|xrow| {
                let input_array: &[Value; 2] = xrow.as_slice().try_into().expect("Input size mismatch");
                model.forward(input_array)[0].clone()
            })
            .collect();

        let losses: Vec<Value> = ys
            .iter()
            .zip(&scores)
            .map(|(yi, scorei)| (Value::from(1.0) + &Value::from(-yi) * scorei).relu())
            .collect();
        let n: f32 = (&losses).len() as f32;
        let data_loss: Value = losses.into_iter().sum::<Value>() / Value::from(n);

        let alpha: f32 = 0.0001;
        let reg_loss: Value = Value::from(alpha) * model.parameters().map(|p| p * p).into_iter().sum::<Value>();
        let total_loss = data_loss + reg_loss;

        total_loss
    }

    let range = 150;
    let mut total_loss: Value = Value::from(0.0);
    // initalized so the checker isnt giving me errors
    for k in 0..range {
        // forward
        total_loss = loss(&x, &y, &model);

        // backward
        model.parameters().for_each(|p| p.zero_grad());

        total_loss.backward();

        // update (sgd)
        let learning_rate = 1.0 - 0.9 * (k as f32) / (range as f32);
        for p in model.parameters() {
            let delta = learning_rate * *p.grad.borrow();
            *p.data.borrow_mut() -= delta;
        }
    }
    assert_eq!(format!("{:.2}", total_loss.data()), "0.01");
}
