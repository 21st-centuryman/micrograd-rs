use crate::engine::Value;
use rand::Rng;

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Neuron {
        Neuron {
            w: (0..nin).map(|_| Value::from(rand::thread_rng().gen_range(-1.0..=1.0))).collect(),
            b: Value::from(0.0), // Removing the bias
            nonlin,
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Value {
        let out = (std::iter::zip(&self.w, x)
            .map(|(wi, xi)| wi * xi)
            .collect::<Vec<_>>()
            .into_iter()
            .reduce(|a, b| a + b)
            .unwrap())
            + self.b.clone();
        return if self.nonlin { out.relu() } else { out };
    }

    pub fn parameters(&self) -> Vec<Value> {
        [self.w.clone(), vec![self.b.clone()]].concat()
    }
}

impl std::fmt::Debug for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}Neuron({})", if self.nonlin { "ReLU" } else { "Linear" }, self.w.len())
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect(),
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|i| i.forward(x)).collect::<Vec<_>>()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|neuron| neuron.parameters()).collect()
    }
}

impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layer of [{:?}]",
            self.neurons.iter().map(|n| format!("{:?}", n)).collect::<Vec<_>>().join(", ")
        )
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> MLP {
        MLP {
            layers: (0..nouts.len())
                .map(|i| {
                    Layer::new(
                        vec![nin].into_iter().chain(nouts.clone().into_iter()).collect::<Vec<_>>()[i],
                        nouts[i],
                        i != nouts.len() - 1,
                    )
                })
                .collect(),
        }
    }

    pub fn forward(&self, x: Vec<Value>) -> Vec<Value> {
        self.layers.iter().fold(x, |x, layer| layer.forward(&x))
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }
}

impl std::fmt::Debug for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MLP of [{}]",
            self.layers.iter().map(|l| format!("{:?}", l)).collect::<Vec<_>>().join(", ")
        )
    }
}
