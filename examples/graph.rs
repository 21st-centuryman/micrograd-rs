use micrograd::engine::{Activations::Relu, Tape};
use micrograd::nn::Layer;
use petgraph::dot::Dot;
use petgraph::graph::Graph;
use std::fs::File;
use std::io::Write;
use std::process::Command;

// The tape is already the graph, every node points back to its inputs
fn draw_dot(tape: &Tape) {
    let mut g = Graph::<String, String>::new();
    let node_ids: Vec<_> = (0..tape.len())
        .map(|i| {
            let n = tape.get(i);
            g.add_node(format!("data {:.4} \ngrad {:.4} ", n.data, n.grad))
        })
        .collect();

    (0..tape.len()).for_each(|i| {
        let n = tape.get(i);
        n.prev.iter().flatten().for_each(|&child| {
            g.add_edge(node_ids[child as usize], node_ids[i as usize], n.op.unwrap_or("").to_string());
        });
    });

    let _ = File::create("graph.dot").unwrap().write_all(Dot::new(&g).to_string().as_bytes()).unwrap();

    let _ = Command::new("dot")
        .arg("-Tsvg")
        .arg("graph.dot")
        .arg("-o")
        .arg("graph.svg")
        .output()
        .expect("Failed to execute Graphviz");
}

fn main() {
    let tape: &Tape = &Tape::<[_; 32]>::new();
    let n = Layer::<2, 1>::new(tape, Relu);
    let x = [tape.value(1.0), tape.value(-2.0)];
    let y = n.forward(&x);
    y[0].backward();
    draw_dot(tape);
}
