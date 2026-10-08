use crate::engine::{Activations, Value};
use std::{
    array::from_fn,
    fmt::{Debug, Formatter, Result},
    iter::once,
};

#[macro_export]
macro_rules! mlp {
    ($layers:literal) => {
        $crate::generate_mlp!($layers);
    };
}

// RNG generator using SplitMix64
// SAFETY: only called while constructing layers, from one thread, never from an interrupt handler.
static mut RNG_STATE: u64 = 1337;

fn uniform(a: f32, b: f32) -> f32 {
    let mut z = unsafe {
        RNG_STATE = RNG_STATE.wrapping_add(0x9E3779B97F4A7C15);
        RNG_STATE
    };
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    a + (b - a) * ((z >> 40) as f32 * (1.0 / (1u32 << 24) as f32))
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
            w: from_fn(|_| from_fn(|_| Value::from(uniform(-1.0, 1.0)))),
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
