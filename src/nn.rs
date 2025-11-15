use crate::engine::{Activations, Value};
use rand::Rng;
use std::{
    array::from_fn,
    fmt::{Debug, Formatter, Result},
    iter::once,
};

#[macro_export]
macro_rules! mlp {
    ($layers:literal) => {
        mlp_macro::generate_mlp!($layers);
    };
}

// Structs
pub struct Layer<const P: usize, const N: usize> {
    w: [[Value; P]; N],
    b: [Value; N],
    nonlin: Activations,
}

// Implementation
impl<const P: usize, const N: usize> Layer<P, N> {
    pub fn new(nonlin: Activations) -> Layer<P, N> {
        Self {
            w: from_fn(|_| from_fn(|_| Value::from(rand::thread_rng().gen_range(-1.0..=1.0)))),
            b: from_fn(|_| Value::from(0.0)),
            nonlin,
        }
    }

    pub fn forward(&self, x: &[Value; P]) -> [Value; N] {
        Value::activate(Value::matmul_add::<P, N>(&self.w, &x, &self.b), &self.nonlin)
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.w.iter().zip(self.b.iter()).flat_map(|(ws, b)| ws.iter().chain(once(b)))
    }
}

// Formater for print out
impl<const P: usize, const N: usize> Debug for Layer<P, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "Layer [{}, {}]",
            match self.nonlin {
                Activations::Relu => "ReLU",
                Activations::Tanh => "Tanh",
                Activations::Linear => "Linear",
            },
            N
        )
    }
}

pub use mlp;
