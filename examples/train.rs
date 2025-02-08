/*
-------------------------------
Now for some actual Machine Learning
--------------------------------
*/
use kdam::{tqdm, BarExt};
use micrograd::engine::Value;
use micrograd::nn::MLP;

fn main() {
    // Variables
    let range = 1000;
    let adjust = -0.01;
    let mut pb = tqdm!(total = range);
    let _ = pb.refresh();
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
        //println!("{k}: Loss: {:.3}", loss.data()); // If we want to print loss
        if k == range - 1 {
            println!("\nACTUAL");
            println!(
                "   [{:?}]",
                ypred.iter().map(|x| format!("{:.3}", x.data())).collect::<Vec<_>>().join(", ")
            );
            println!("DESIRED");
            println!("   [{:?}]", ys.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>().join(", "));
        }
        pb.set_description(format!("Loss: {:.5}", loss.data()));
        let _ = pb.update(1);
    }
}
