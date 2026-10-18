/*
----------------------------------------------------------------------------------
This is a test to make sure that the tests in the examples pass.
----------------------------------------------------------------------------------
*/

use micrograd::{
    engine::{
        Activations::{Linear, Relu},
        Tape, Value,
    },
    nn::mlp,
};

#[test]
pub fn test_usage() {
    let tape: &Tape = &Tape::<[_; 64]>::new();
    let a = tape.value(-4.0);
    let b = tape.value(2.0);

    let mut c = a + b;
    let mut d = a * b + b.pow(3.0);
    c = 2.0 * c + 1.0;
    c = 1.0 + 2.0 * c + (-a);
    d = d + d * 2.0 + (b + a).relu();
    d = d + 3.0 * d + (b - a).relu();
    let e = c - d;
    let f = e.pow(2.0);
    let mut g = f / 2.0;
    g = g + 10.0 / f;

    assert_eq!(format!("{:.4}", g.data()), "24.7041");
    g.backward();
    assert_eq!(format!("{:.4}", a.grad()), "138.8338");
    assert_eq!(format!("{:.4}", b.grad()), "645.5773");
}

#[test]
fn train() {
    mlp!(3, 4, 4, 1);
    let tape: &Tape = &MLPTape::new();
    let n = MLP::new(tape, Relu, Relu, Linear);
    let params = tape.len.get();

    // Variables
    let range = 2000;
    let adjust = -0.01;
    let ys = vec![1.0, -1.0, -1.0, 1.0]; // desired targets

    let xs = vec![vec![2.0, 3.0, -1.0], vec![3.0, -1.0, 0.5], vec![0.5, 1.0, 1.0], vec![1.0, 1.0, -1.0]];

    let mut ypred = [0.0; 4];
    for k in 0..range {
        // Backward pass
        for p in n.parameters() {
            p.zero_grad();
        }

        // Forward + backward
        for (i, (x, ygt)) in xs.iter().zip(&ys).enumerate() {
            let yout = n.forward(&[tape.value(x[0]), tape.value(x[1]), tape.value(x[2])])[0];
            let loss = (yout - tape.value(*ygt)).pow(2.0);
            loss.backward();
            ypred[i] = yout.data();
            tape.len.set(params);
        }

        // Update
        for p in n.parameters() {
            p.adjust(adjust);
        }

        if k == range - 1 {
            assert_eq!(format!("{:.3}", ypred[0]), "1.000");
            assert_eq!(format!("{:.3}", ypred[1]), "-1.000");
            assert_eq!(format!("{:.3}", ypred[2]), "-1.000");
            assert_eq!(format!("{:.3}", ypred[3]), "1.000");
        }
    }
}

#[test]
fn make_moons() {
    mlp!(2, 16, 16, 1);
    let tape: &Tape = &MLPTape::new();
    let model = MLP::new(tape, Relu, Relu, Linear);

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

    fn loss<'t>(tape: &'t Tape, xs: &[[f32; 2]], ys: &[f32], model: &MLP<'t>) -> f32 {
        let params = tape.len.get();
        let n = xs.len() as f32;
        let mut total = 0.0;

        // Data loss
        for (xrow, yi) in xs.iter().zip(ys) {
            let score = model.forward(&[tape.value(xrow[0]), tape.value(xrow[1])])[0];
            let loss = (tape.value(1.0) + tape.value(-yi) * score).relu() / tape.value(n);
            loss.backward();
            total += loss.data();
            tape.len.set(params);
        }

        // Regularization
        let alpha: f32 = 0.0001;
        let reg_loss: Value = tape.value(alpha) * model.parameters().map(|p| p * p).sum::<Value>();
        reg_loss.backward();
        total += reg_loss.data();
        tape.len.set(params);

        total
    }

    let range = 150;
    let mut total_loss = 0.0;
    for k in 0..range {
        // Backward pass
        model.parameters().for_each(|p| p.zero_grad());

        total_loss = loss(tape, &x, &y, &model);

        // update (sgd)
        let learning_rate = 1.0 - 0.9 * (k as f32) / (range as f32);
        for p in model.parameters() {
            p.adjust(-learning_rate);
        }
    }
    assert!(total_loss < 0.20);
}
