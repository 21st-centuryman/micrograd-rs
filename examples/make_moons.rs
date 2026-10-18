use csv;
use kdam::{BarExt, tqdm};
use micrograd::{
    engine::{
        Activations::{Linear, Relu},
        Tape, Value,
    },
    nn::mlp,
};

// Initialize model size
mlp!(2, 16, 16, 1);

fn main() {
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
    let mut pb = tqdm!(total = range);
    let _ = pb.refresh();
    for k in 0..range {
        // backward
        model.parameters().for_each(|p| p.zero_grad());

        let total_loss = loss(tape, &x, &y, &model);

        // update (sgd)
        let learning_rate = 1.0 - 0.9 * (k as f32) / (range as f32);
        for p in model.parameters() {
            p.adjust(-learning_rate);
        }

        pb.set_description(format!("Loss {:.3}", total_loss));
        let _ = pb.update(1);
    }
}
