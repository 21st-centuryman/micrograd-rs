use crate::engine::Value;
use rand::Rng;

pub struct Layer<const P: usize, const N: usize> {
    w: [[Value; P]; N],
    b: [Value; N],
    nonlin: bool,
}

pub struct MLP<const N1: usize, const N2: usize, const N3: usize, const N4: usize> {
    l1: Layer<N1, N2>,
    l2: Layer<N2, N3>,
    l3: Layer<N3, N4>,
}

impl<const P: usize, const N: usize> Layer<P, N> {
    pub fn new(nonlin: bool) -> Layer<P, N> {
        Self {
            w: std::array::from_fn(|_| std::array::from_fn(|_| Value::from(rand::thread_rng().gen_range(-1.0..=1.0)))),
            b: std::array::from_fn(|_| Value::from(0.0)),
            nonlin,
        }
    }

    pub fn forward(&self, x: &[Value; P]) -> [Value; N] {
        std::array::from_fn(|i| {
            let out = self.w[i].iter().zip(x.iter()).map(|(w, x)| w * x).reduce(|a, b| a + b).unwrap() + self.b[i].clone();
            return if self.nonlin { out.relu() } else { out };
        })
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.w.iter().zip(self.b.iter()).flat_map(|(ws, b)| ws.iter().chain(std::iter::once(b)))
    }
}

impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize> MLP<N1, N2, N3, N4> {
    pub fn new() -> Self {
        Self {
            l1: Layer::<N1, N2>::new(true),
            l2: Layer::<N2, N3>::new(true),
            l3: Layer::<N3, N4>::new(false),
        }
    }

    pub fn forward(&self, x: &[Value; N1]) -> [Value; N4] {
        self.l3.forward(&self.l2.forward(&self.l1.forward(&x)))
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.l1.parameters().chain(self.l2.parameters()).chain(self.l3.parameters())
    }
}

impl<const P: usize, const N: usize> std::fmt::Debug for Layer<P, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer [{}, {}]", if self.nonlin { "ReLU" } else { "Linear" }, N)
    }
}

impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize> std::fmt::Debug for MLP<N1, N2, N3, N4> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MLP of [{:?}, {:?}, {:?}]", self.l1, self.l2, self.l3)
    }
}
