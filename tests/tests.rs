/*
----------------------------------------------------------------------------------
This is a test to make sure that the tests in the examples pass.
----------------------------------------------------------------------------------
*/

use csv;
use micrograd::engine::Value;
use micrograd::nn::MLP;

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

    assert_eq!(format!("{:.4}", g.borrow().data), "24.7041");
    g.backward();
    assert_eq!(format!("{:.4}", a.borrow().grad), "145.7755");
    assert_eq!(format!("{:.4}", b.borrow().grad), "645.5773");
}

#[test]
fn train() {
    // Variables
    let range = 2000;
    let adjust = -0.01;
    let ys = vec![1.0, -1.0, -1.0, 1.0]; // desired targets

    let n = MLP::new(3, vec![4, 4, 1]);

    let xs = vec![vec![2.0, 3.0, -1.0], vec![3.0, -1.0, 0.5], vec![0.5, 1.0, 1.0], vec![1.0, 1.0, -1.0]];

    for k in 0..range {
        // Forward pass
        let ypred: Vec<Value> = xs
            .iter()
            .map(|x| n.forward(x.iter().map(|x| Value::from(*x)).collect())[0].clone())
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
fn gradient_decent() {
    let path = "./datasets/make_moons.csv";
    let x: Vec<[f64; 2]> = csv::ReaderBuilder::new()
        .from_path(path)
        .unwrap()
        .records()
        .map(|r| {
            let record = r.unwrap();
            [
                record.get(0).unwrap().parse::<f64>().unwrap(),
                record.get(1).unwrap().parse::<f64>().unwrap(),
            ]
        })
        .collect();
    let y: Vec<f64> = csv::ReaderBuilder::new()
        .from_path(path)
        .unwrap()
        .records()
        .map(|r| r.unwrap().get(2).unwrap().parse::<f64>().unwrap())
        .collect();

    let model = MLP::new(2, vec![10, 10, 1]);

    fn loss(xs: Vec<[f64; 2]>, ys: Vec<f64>, model: &MLP) -> (Value, f64) {
        let inputs: Vec<Vec<Value>> = xs.iter().map(|xrow| vec![Value::from(xrow[0]), Value::from(xrow[1])]).collect();

        let scores: Vec<Value> = inputs.iter().map(|xrow| model.forward(xrow.clone())[0].clone()).collect();

        let losses: Vec<Value> = ys
            .iter()
            .zip(&scores)
            .map(|(yi, scorei)| (Value::from(1.0) + &Value::from(-yi) * scorei).relu())
            .collect();
        let n: f64 = (&losses).len() as f64;
        let data_loss: Value = losses.into_iter().sum::<Value>() / Value::from(n);

        let alpha: f64 = 0.0001;
        let reg_loss: Value = Value::from(alpha) * model.parameters().iter().map(|p| p * p).into_iter().sum::<Value>();
        let total_loss = data_loss + reg_loss;

        let accuracies: Vec<bool> = ys
            .iter()
            .zip(scores.iter())
            .map(|(yi, scorei)| (*yi > 0.0) == (scorei.borrow().data > 0.0))
            .collect();
        let accuracy = accuracies.iter().filter(|&a| *a).count() as f64 / n;

        (total_loss, accuracy)
    }

    let range = 70;
    for k in 0..range {
        let (total_loss, acc) = loss(x.clone(), y.clone(), &model);

        model.parameters().iter().for_each(|p| p.zero_grad());

        total_loss.backward();

        let learning_rate = 1.0 - 0.9 * (k as f64) / 100.0;
        for p in &model.parameters() {
            let delta = learning_rate * p.borrow().grad;
            p.borrow_mut().data -= delta;
        }
        if k == range-1 {
            assert_eq!(format!("{:.2}", acc * 100.0), "100.00");
            assert_eq!(format!("{:.2}", total_loss.borrow().data), "0.01");
        }
    }
}
