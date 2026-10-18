use std::{
    array::from_fn,
    cell::Cell,
    fmt::{Debug, Formatter, Result},
    iter::Sum,
};

#[derive(Clone, Copy)]
pub struct ValueData {
    pub data: f32,
    pub grad: f32,
    pub op: Option<&'static str>,
    pub prev: [Option<u32>; 2],
    pub _backward: Option<fn(&Tape, u32)>,
}

#[derive(Clone, Copy)]
pub struct Value<'t> {
    tape: &'t Tape,
    idx: u32,
}

pub struct Tape<S: ?Sized = [Cell<ValueData>]> {
    pub len: Cell<u32>,
    nodes: S,
}

pub enum Activations {
    Linear,
    Relu,
    Tanh,
}

impl<const L: usize> Tape<[Cell<ValueData>; L]> {
    pub const fn new() -> Self {
        Tape {
            len: Cell::new(0),
            nodes: [const { Cell::new(ValueData::EMPTY) }; L],
        }
    }
}

impl Tape {
    pub fn len(&self) -> u32 {
        self.len.get()
    }

    pub fn value(&self, data: f32) -> Value<'_> {
        self.push(ValueData { data, ..ValueData::EMPTY })
    }

    fn push(&self, node: ValueData) -> Value<'_> {
        let idx = self.len.replace(self.len.get() + 1);
        self.nodes[idx as usize].set(node);
        Value { tape: self, idx }
    }

    pub fn get(&self, i: u32) -> ValueData {
        self.nodes[i as usize].get()
    }

    fn add_grad(&self, i: u32, g: f32) {
        self.nodes[i as usize].update(|n| ValueData { grad: n.grad + g, ..n });
    }

    fn set_grad(&self, i: u32, g: f32) {
        self.nodes[i as usize].update(|n| ValueData { grad: g, ..n });
    }
}

impl ValueData {
    const fn new(data: f32, op: Option<&'static str>, prev: [Option<u32>; 2], _backward: Option<fn(&Tape, u32)>) -> ValueData {
        ValueData {
            data,
            grad: 0.0,
            op,
            prev,
            _backward,
        }
    }

    pub const EMPTY: ValueData = ValueData::new(0.0, None, [None, None], None);
}

impl<'t> Value<'t> {
    pub fn data(&self) -> f32 {
        self.tape.get(self.idx).data
    }

    pub fn grad(&self) -> f32 {
        self.tape.get(self.idx).grad
    }

    pub fn zero_grad(&self) {
        self.tape.set_grad(self.idx, 0.0);
    }

    pub fn adjust(&self, val: f32) {
        self.tape.nodes[self.idx as usize].update(|n| ValueData {
            data: n.data + val * n.grad,
            ..n
        })
    }

    pub fn div(a: Value<'t>, b: Value<'t>) -> Value<'t> {
        Value::mul(a, b.powneg())
    }

    pub fn pow(self, k: f32) -> Value<'t> {
        Value::pow_op(self, self.tape.value(k))
    }

    pub fn matmul<const M: usize, const N: usize>(a: &[[Value<'t>; N]; M], b: &[Value<'t>; N]) -> [Value<'t>; M] {
        from_fn(|i| a[i].iter().zip(b.iter()).map(|(a, b)| a * b).reduce(|a, b| a + b).unwrap())
    }

    pub fn matadd<const N: usize, const M: usize>(a: &[[Value<'t>; N]; M], b: &[[Value<'t>; N]; M]) -> [[Value<'t>; N]; M] {
        from_fn(|i| from_fn(|j| &a[i][j] + &b[i][j]))
    }

    pub fn matmul_add<const P: usize, const N: usize>(a: &[[Value<'t>; P]; N], b: &[Value<'t>; P], c: &[Value<'t>; N]) -> [Value<'t>; N] {
        from_fn(|i| &a[i].iter().zip(b.iter()).map(|(a, b)| a * b).reduce(|a, b| a + b).unwrap() + &c[i])
    }

    pub fn activate<const I: usize>(a: [Value<'t>; I], b: &Activations) -> [Value<'t>; I] {
        match b {
            Activations::Linear => a,
            Activations::Tanh => from_fn(|i| a[i].tanh()),
            Activations::Relu => from_fn(|i| a[i].relu()),
        }
    }

    pub fn sum<I: Iterator<Item = Value<'t>>>(iter: I) -> Value<'t> {
        iter.reduce(|a, b| a + b)
            .expect("cannot sum an empty iterator: there is no tape to put a zero on")
    }

    pub fn backward(&self) {
        self.tape.set_grad(self.idx, 1.0);
        (0..=self.idx).rev().for_each(|i| {
            if let Some(f) = self.tape.get(i)._backward {
                f(self.tape, i);
            }
        });
    }
}

macro_rules! define_ops {
    (
        $(binary $bname:ident, $bsym:expr => |$a:ident, $b:ident| $bfwd:expr, |$tb:ident, $out:ident| $bbwd:block;)*
        $(unary $uname:ident, $usym:expr => |$x:ident| $ufwd:expr, |$tu:ident, $outu:ident| $ubwd:block;)*
    ) => {
        $(
            impl<'t> Value<'t> {
                pub fn $bname($a: Value<'t>, $b: Value<'t>) -> Value<'t> {
                    debug_assert!(core::ptr::eq($a.tape, $b.tape), "values from different tapes");
                    let _backward: fn(&Tape, u32) = |$tb, $out| $bbwd;
                    $a.tape.push(ValueData::new($bfwd, Some($bsym), [Some($a.idx), Some($b.idx)], Some(_backward)))
                }
            }
        )*
        $(
            impl<'t> Value<'t> {
                pub fn $uname(self) -> Value<'t> {
                    let _backward: fn(&Tape, u32) = |$tu, $outu| $ubwd;
                    self.tape.push(ValueData::new({ let $x = self; $ufwd }, Some($usym), [Some(self.idx), None], Some(_backward)))
                }
            }
        )*
    };
}

define_ops! {
    binary add, "+" => |a, b| a.data() + b.data(), |t, out| {
        let o = t.get(out);
        t.add_grad(o.prev[0].unwrap(), o.grad);
        t.add_grad(o.prev[1].unwrap(), o.grad);
    };

    binary sub, "-" => |a, b| a.data() - b.data(), |t, out| {
        let o = t.get(out);
        t.add_grad(o.prev[0].unwrap(), o.grad);
        t.add_grad(o.prev[1].unwrap(), -o.grad);
    };

    binary mul, "*" => |a, b| a.data() * b.data(), |t, out| {
        let o = t.get(out);
        let (a, b) = (o.prev[0].unwrap(), o.prev[1].unwrap());
        let (a_data, b_data) = (t.get(a).data, t.get(b).data);
        t.add_grad(a, b_data * o.grad);
        t.add_grad(b, a_data * o.grad);
    };

    binary pow_op, "^" => |a, b| a.data().powf(b.data()), |t, out| {
        let o = t.get(out);
        let (a, b) = (o.prev[0].unwrap(), o.prev[1].unwrap());
        let (base, exp) = (t.get(a).data, t.get(b).data);
        t.add_grad(a, exp * base.powf(exp - 1.0) * o.grad);
    };

    unary powneg, "^-" => |x| 1.0 / x.data(), |t, out| {
        let o = t.get(out);
        let x = o.prev[0].unwrap();
        let x_data = t.get(x).data;
        t.add_grad(x, -o.grad / (x_data * x_data));
    };

    unary exp, "exp" => |x| x.data().exp(), |t, out| {
        let o = t.get(out);
        t.add_grad(o.prev[0].unwrap(), o.data * o.grad);
    };

    unary ln, "log" => |x| x.data().ln(), |t, out| {
        let o = t.get(out);
        let x = o.prev[0].unwrap();
        t.add_grad(x, o.grad / t.get(x).data);
    };

    unary tanh, "tanh" => |x| x.data().tanh(), |t, out| {
        let o = t.get(out);
        t.add_grad(o.prev[0].unwrap(), (1.0 - o.data * o.data) * o.grad);
    };

    unary relu, "ReLU" => |x| x.data().max(0.0), |t, out| {
        let o = t.get(out);
        if o.data > 0.0 { t.add_grad(o.prev[0].unwrap(), o.grad); }
    };
}

impl Debug for Value<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "Value(data={}, grad={})", self.data(), self.grad())
    }
}

/*
------------------------------------------------------------------------------------------------
This allows us to use Value + Value instead of Value.add(Value), it works just like micrograd
------------------------------------------------------------------------------------------------
*/

macro_rules! impl_ops {
    ( $( $Trait:ident::$method:ident ),* $(,)? ) => {
        $(
            impl<'t> core::ops::$Trait<Value<'t>> for Value<'t> {
                type Output = Value<'t>;
                fn $method(self, other: Value<'t>) -> Value<'t> { Value::$method(self, other) }
            }
            impl<'a, 't> core::ops::$Trait<&'a Value<'t>> for &'a Value<'t> {
                type Output = Value<'t>;
                fn $method(self, other: &'a Value<'t>) -> Value<'t> { Value::$method(*self, *other) }
            }
            impl<'t> core::ops::$Trait<f32> for Value<'t> {
                type Output = Value<'t>;
                fn $method(self, k: f32) -> Value<'t> { Value::$method(self, self.tape.value(k)) }
            }
            impl<'t> core::ops::$Trait<Value<'t>> for f32 {
                type Output = Value<'t>;
                fn $method(self, v: Value<'t>) -> Value<'t> { Value::$method(v.tape.value(self), v) }
            }
        )*
    }
}

impl_ops!(Add::add, Sub::sub, Mul::mul, Div::div);

impl<'t> core::ops::Neg for Value<'t> {
    type Output = Value<'t>;
    fn neg(self) -> Value<'t> {
        self * -1.0
    }
}

impl<'a, 't> core::ops::Neg for &'a Value<'t> {
    type Output = Value<'t>;
    fn neg(self) -> Value<'t> {
        *self * -1.0
    }
}

impl<'t> Sum for Value<'t> {
    fn sum<I: Iterator<Item = Value<'t>>>(iter: I) -> Value<'t> {
        Value::sum(iter)
    }
}
