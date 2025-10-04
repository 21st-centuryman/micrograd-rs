use crate::engine::Value;
use rand::Rng;

// MLP Macros
macro_rules! forward {
    ($x:expr)                      => { $x };
    ($outer:expr, $($rest:expr),+) => { $outer.forward(&forward!($($rest),+)) };
}
macro_rules! parameters {
    ($layer:expr)                  => { $layer.parameters() };
    ($layer:expr, $($rest:expr),+) => { $layer.parameters().chain(parameters!($($rest),+)) };
}

// Structs
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

// Implementation
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

    //pub fn parameters(&self) -> [Value; P + 1] {
    //    std::array::from_fn(|i| self.w.get(i).cloned().unwrap_or_else(|| self.b.clone()))
    //}
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
        forward!(self.l3, self.l2, self.l1, x)
    }
    pub fn parameters(&self) -> impl Iterator<Item = &Value> {
        parameters!(self.l1, self.l2, self.l3)
    }

    //pub fn parameters(&self) -> [Value; N * (P + 1)] {
    //    std::array::from_fn(|i| self.neurons[i / (P + 1)].parameters()[i % (P + 1)].clone())
    //}
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

//#[macro_export]
//macro_rules! mlp {
//    ($n1:expr, $n2:expr, $n3:expr, $n4:expr) => {
//        pub struct MLP<const N1: usize, const N2: usize, const N3: usize, const N4: usize> {
//            l1: Layer<N1, N2>,
//            l2: Layer<N2, N3>,
//            l3: Layer<N3, N4>,
//        }
//
//        impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize> MLP<N1, N2, N3, N4> {
//            pub fn new() -> Self {
//                Self {
//                    l1: Layer::new(true),
//                    l2: Layer::new(true),
//                    l3: Layer::new(false),
//                }
//            }
//
//            pub fn forward(&self, x: &[Value; N1]) -> [Value; N4] {
//                forward!(self.l3, self.l2, self.l1, x)
//            }
//            pub fn parameters(&self) -> impl Iterator<Item = &Value> {
//                parameters!(self.l1, self.l2, self.l3)
//            }
//        }
//        impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize> std::fmt::Debug for MLP<N1, N2, N3, N4> {
//            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//                write!(f, "MLP of [{:?}, {:?}, {:?}]", self.l1, self.l2, self.l3)
//            }
//        }
//    };
//    ($n1:expr, $n2:expr, $n3:expr) => {
//        pub struct MLP<const N1: usize, const N2: usize, const N3: usize> {
//        l1: Layer<N1, N2>,
//        l2: Layer<N2, N3>,
//    }
//
//        impl<const N1: usize, const N2: usize, const N3: usize> MLP<N1, N2, N3> {
//        pub fn new() -> Self {
//    Self {
//    l1: Layer::new(true),
//    l2: Layer::new(false),
//    }
//            }
//
//            pub fn forward(&self, x: &[Value; N1]) -> [Value; N3] {
//                forward!(self.l2, self.l1, x)
//            }
//            pub fn parameters(&self) -> impl Iterator<Item = &Value> {
//                parameters!(self.l1, self.l2)
//            }
//        }
//        impl<const N1: usize, const N2: usize, const N3: usize: usize> std::fmt::Debug for MLP<N1, N2, N3> {
//            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//                write!(f, "MLP of [{:?}, {:?}, {:?}]", self.l1, self.l2)
//            }
//        }
//    };
//}
