use crate::engine::Value;
use rand::Rng;

pub struct Neuron<const P: usize> {
    w: [Value; P],
    b: Value,
    nonlin: bool,
}

impl<const P: usize> Neuron<P> {
    pub fn new(nonlin: bool) -> Self {
        Neuron {
            w: std::array::from_fn(|_| Value::from(rand::thread_rng().gen_range(-1.0..=1.0))),
            b: Value::from(0.0), // Removing the bias
            nonlin,
        }
    }

    pub fn forward(&self, x: &[Value; P]) -> Value {
        let out = match P {
            1 => (&self.w[0] * &x[0]) + self.b.clone(),
            2 => (&self.w[0] * &x[0] + &self.w[1] * &x[1]) + self.b.clone(),
            _ => {
                let mut sum = Value::from(0.0);
                for (wi, xi) in self.w.iter().zip(x.iter()) {
                    sum = sum + (wi * xi);
                }
                sum + self.b.clone()
            }
        };
        return if self.nonlin { out.relu() } else { out };
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.w.iter().chain(std::iter::once(&self.b))
    }

    //pub fn parameters(&self) -> [Value; P + 1] {
    //    std::array::from_fn(|i| self.w.get(i).cloned().unwrap_or_else(|| self.b.clone()))
    //}
}

impl<const N: usize> std::fmt::Debug for Neuron<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}Neuron({})", if self.nonlin { "ReLU" } else { "Linear" }, N)
    }
}

pub struct Layer<const P: usize, const N: usize> {
    neurons: [Neuron<P>; N],
}

impl<const P: usize, const N: usize> Layer<P, N> {
    pub fn new(nonlin: bool) -> Layer<P, N> {
        Self {
            neurons: std::array::from_fn(|_| Neuron::new(nonlin)),
        }
    }

    pub fn forward(&self, x: &[Value; P]) -> [Value; N] {
        std::array::from_fn(|i| self.neurons[i].forward(x))
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        self.neurons.iter().flat_map(|n| n.parameters())
    }
    //pub fn parameters(&self) -> [Value; N * (P + 1)] {
    //    std::array::from_fn(|i| self.neurons[i / (P + 1)].parameters()[i % (P + 1)].clone())
    //}
}
impl<const P: usize, const N: usize> std::fmt::Debug for Layer<P, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layer of [{:?}]",
            self.neurons.iter().map(|n| format!("{:?}", n)).collect::<Vec<_>>().join(", ")
        )
    }
}

pub struct MLP<const N1: usize, const N2: usize, const N3: usize, const N4: usize> {
    l1: Layer<N1, N2>,
    l2: Layer<N2, N3>,
    l3: Layer<N3, N4>,
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

    //pub fn parameters(&self) -> [Value; N2 * (N1 + 1) + N3 * (N2 + 1) + N4 * (N3 + 1)] {
    //    let seg1 = N2 * (N1 + 1); // Layer 1 end
    //    let seg2 = seg1 + N3 * (N2 + 1); // Layer 2 end

    //    std::array::from_fn(|i| match i {
    //        i if i < seg1 => self.l1.parameters()[i].clone(),
    //        i if i < seg2 => self.l2.parameters()[i - seg1].clone(),
    //        _ => self.l3.parameters()[i - seg2].clone(),
    //    })
    //}
}

impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize> std::fmt::Debug for MLP<N1, N2, N3, N4> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MLP of [{:?}, {:?}, {:?}]", self.l1, self.l2, self.l3)
    }
}
