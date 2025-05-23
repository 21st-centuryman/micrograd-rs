<div align="center">

## Micrograd
[![RUST](https://img.shields.io/badge/rust-f74c00.svg?style=for-the-badge&logoColor=white&logo=rust)]()
[![EVCXR](https://img.shields.io/badge/Evcxr_notebook-F37626.svg?style=for-the-badge&logoColor=white&logo=jupyter)]()
<br>
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/jay-lex/micrograd/main.yml?branch=main&style=for-the-badge&logo=github&logoColor=white&label=Test&labelColor=black)

![awww](assets/crab.png)
</div>

## ⇁  Welcome
This is a thesis project to rewrite the [micrograd](https://github.com/karpathy/micrograd) framework by Andrej Karpathy to Rust. The purpose of this thesis is to discuss memory allocation of micrograd when rewritten in the Rust programming language. I looked over the limitations of writing the project similarly to Rust and how it can be improved to focus on stack allocation (which I will implement in the micrograd_v2). The report will be added to this project once it is published.

## ⇁  Example usage
```rs
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
println!("{:.4}", g.borrow().data); // prints 24.7041
g.backward();
println!("{:.4}", a.borrow().grad); // print 138.8338
println!("{:.4}", b.borrow().grad); // print 645.5773
```

## ⇁  Tracing / Visualization

We also implemented a draw_dot function that will act similarly to graphviz and digraph. This will allow us to visualize each node, showing both their data and gradient. An example below shows how to run it. Also `trace_graph.ipynb` shows how to achieve this using [evxcr jupyter kernel](https://github.com/evcxr/evcxr/blob/main/evcxr_jupyter/README.md)
```rust
use micrograd::engine::Value;
use micrograd::nn::Neuron;
let x = Value::from(1.0);
let y = (x * Value::from(2) + Value::from(1)).relu();
y.backward();
draw_dot(y);
```
![2d neuron](assets/graph.svg)

## ⇁  Running tests
All tests are in the `tests` folder. You can run them with the following command.
```console
cargo test
```
## ⇁  License
MIT
</div>
