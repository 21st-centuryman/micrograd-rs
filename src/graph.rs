// This is literally here to be able to visualize our results using a Jupyter notebook
// So we are just saving it

use crate::engine::Value;
use petgraph::dot::Dot;
use petgraph::graph::Graph;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::process::Command;

fn build(v: Value, nodes: &mut HashSet<Value>, edges: &mut HashSet<(Value, Value)>) {
    if !nodes.contains(&v) {
        nodes.insert(v.clone());
        v.borrow().prev.iter().for_each(|child| {
            edges.insert((child.clone(), v.clone()));
            build(child.clone(), nodes, edges);
        });
    }
}

fn trace(root: Value) -> (HashSet<Value>, HashSet<(Value, Value)>) {
    let mut nodes = HashSet::new();
    let mut edges = HashSet::new();

    build(root, &mut nodes, &mut edges);
    (nodes, edges)
}

pub fn draw_dot(root: Value) {
    let (nodes, edges) = trace(root);
    let mut g = Graph::<String, String>::new();
    let node_ids = nodes
        .iter()
        .map(|n| {
            (
                n.clone(),
                g.add_node(format!("data {:.4} \ngrad {:.4} ", n.borrow().data, n.borrow().grad)),
            )
        })
        .collect::<Vec<_>>();

    edges.iter().for_each(|(n1, n2)| {
        let node_id1 = node_ids.iter().find(|(n, _)| n == n1).unwrap().1;
        let node_id2 = node_ids.iter().find(|(n, _)| n == n2).unwrap().1;
        g.add_edge(node_id1, node_id2, n2.borrow().op.clone().expect("REASON"));
    });

    let _ = File::create("graph.dot").unwrap().write_all(Dot::new(&g).to_string().as_bytes()).unwrap();

    let _ = Command::new("dot")
        .arg("-Tsvg")
        .arg("graph.dot")
        .arg("-o")
        .arg("graph.svg")
        .output()
        .expect("Failed to execute Graphviz");

    // Read the SVG file
    let _ = std::fs::read_to_string("graph.svg").unwrap();
}
