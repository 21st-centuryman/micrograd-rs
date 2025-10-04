use csv;
use kdam::{tqdm, BarExt};
use micrograd::engine::{Activations, Value};
//use micrograd::mlp;
use micrograd::nn::{mlp, Layer};

// Initialize model size
mlp!(4);

fn main() {
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
    let mut pb = tqdm!(total = range);
    let _ = pb.refresh();
    for k in 0..range {
        // forward
        let total_loss = loss(&x, &y, &model);

        // backward
        model.parameters().for_each(|p| p.zero_grad());

        total_loss.backward();

        // update (sgd)
        let learning_rate = 1.0 - 0.9 * (k as f32) / (range as f32);
        for p in model.parameters() {
            let delta = learning_rate * *p.grad.borrow();
            *p.data.borrow_mut() -= delta;
        }

        pb.set_description(format!("Loss {:.3}", total_loss.data.borrow()));
        let _ = pb.update(1);
    }
}
