use crate::engine::{Activations, Tape, Value};
use std::{
    array::from_fn,
    fmt::{Debug, Formatter, Result},
    iter::once,
    sync::atomic::{AtomicU64, Ordering},
};

#[macro_export]
macro_rules! mlp {
    ($($size:literal),+ $(,)?) => {
        $crate::generate_mlp!($crate; $($size),+);
    };
}

// RNG generator using SplitMix64
static RNG_STATE: AtomicU64 = AtomicU64::new(1337);

fn uniform(a: f32, b: f32) -> f32 {
    const GAMMA: u64 = 0x9E3779B97F4A7C15;
    let mut z = RNG_STATE.fetch_add(GAMMA, Ordering::Relaxed).wrapping_add(GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    a + (b - a) * ((z >> 40) as f32 * (1.0 / (1u32 << 24) as f32))
}

// Structs
pub struct Layer<'t, const P: usize, const N: usize> {
    w: [[Value<'t>; P]; N],
    b: [Value<'t>; N],
    nonlin: Activations,
}

// Implementation
impl<'t, const P: usize, const N: usize> Layer<'t, P, N> {
    pub fn new(tape: &'t Tape, nonlin: Activations) -> Layer<'t, P, N> {
        Self {
            w: from_fn(|_| from_fn(|_| tape.value(uniform(-1.0, 1.0)))),
            b: from_fn(|_| tape.value(0.0)),
            nonlin,
        }
    }

    pub fn forward(&self, x: &[Value<'t>; P]) -> [Value<'t>; N] {
        Value::activate(Value::matmul_add::<P, N>(&self.w, &x, &self.b), &self.nonlin)
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value<'t>> {
        self.w.iter().zip(self.b.iter()).flat_map(|(ws, b)| ws.iter().chain(once(b)))
    }
}

// Formater for print out
impl<const P: usize, const N: usize> Debug for Layer<'_, P, N> {
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
