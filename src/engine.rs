use std::{
    array::from_fn,
    cell::Cell,
    collections::HashSet,
    fmt::{Debug, Formatter, Result},
    hash::{Hash, Hasher},
    iter::Sum,
    ops,
    rc::Rc,
};

#[derive(Clone)]
pub struct Value(Rc<ValueData>);

pub struct ValueData {
    pub data: Cell<f32>,
    pub grad: Cell<f32>,
    pub op: Option<&'static str>,
    pub prev: [Option<Value>; 2],
    pub _backward: Option<fn(value: &Value)>,
}

pub enum Activations {
    Linear,
    Relu,
    Tanh,
}

impl ValueData {
    fn new(data: f32, op: Option<&'static str>, prev: [Option<Value>; 2], _backward: Option<fn(value: &Value)>) -> ValueData {
        ValueData {
            data: Cell::new(data),
            grad: Cell::new(0.0),
            op,
            prev,
            _backward,
        }
    }
}

macro_rules! define_ops {
    (
        $(binary $bname:ident, $bsym:expr => |$a:ident, $b:ident| $bfwd:expr, |$out:ident| $bbwd:block;)*
        $(unary $uname:ident, $usym:expr => |$x:ident| $ufwd:expr, |$outu:ident| $ubwd:block;)*
    ) => {
        $(
            impl Value {
                pub fn $bname($a: &Value, $b: &Value) -> Value {
                    let _backward: fn(&Value) = |$out| $bbwd;
                    Value::new(ValueData::new(
                        $bfwd,
                        Some($bsym),
                        [Some($a.clone()), Some($b.clone())],
                        Some(_backward),
                    ))
                }
            }
        )*
        $(
            impl Value {
                pub fn $uname(&self) -> Value {
                    let _backward: fn(&Value) = |$outu| $ubwd;
                    Value::new(ValueData::new(
                        { let $x = self; $ufwd },
                        Some($usym),
                        [Some(self.clone()), None],
                        Some(_backward),
                    ))
                }
            }
        )*
    };
}

define_ops! {
    binary add, "+" => |a,b| a.0.data.get() + b.0.data.get(), |out| {
        out.0.prev[0].as_ref().unwrap().grad.update(|v| v + out.grad.get());
        out.0.prev[1].as_ref().unwrap().grad.update(|v| v + out.grad.get());
    };
    binary mul, "*" => |a,b| a.0.data.get() * b.0.data.get(), |out| {
        let a_data = out.0.prev[0].as_ref().unwrap().data.get();
        let b_data = out.0.prev[1].as_ref().unwrap().data.get();
        out.0.prev[0].as_ref().unwrap().grad.update(|v| v + b_data * out.0.grad.get());
        out.0.prev[1].as_ref().unwrap().grad.update(|v| v + a_data * out.0.grad.get());
    };
    binary pow_op, "^" => |a,b| a.0.data.get().powf(b.0.data.get()), |out| {
        let base = out.0.prev[0].as_ref().unwrap().data.get();
        let exp  = out.0.prev[1].as_ref().unwrap().data.get();
        let gout = out.0.grad.get();
        let y    = out.0.data.get();
        out.0.prev[0].as_ref().unwrap().grad.update(|v| v + exp * base.powf(exp - 1.0) * gout);
        out.0.prev[1].as_ref().unwrap().grad.update(|v| v + y * base.ln() * gout);
    };
    unary powneg, "^-" => |x| 1.0 / x.0.data.get(), |out| {
        let base = out.0.prev[0].as_ref().unwrap();
        base.grad.update(|g| g - out.grad.get() / (base.data.get() * base.data.get()));
    };
    unary exp, "exp" => |x| x.0.data.get().exp(), |out| {
        out.0.prev[0].as_ref().unwrap().grad.update(|v| v + out.0.data.get() * out.0.grad.get());
    };
    unary ln, "log" => |x| x.0.data.get().ln(), |out| {
        let xv = out.0.prev[0].as_ref().unwrap().data.get();
        out.0.prev[0].as_ref().unwrap().grad.update(|v| v + out.0.grad.get() / xv);
    };
    unary tanh, "tanh" => |x| x.0.data.get().tanh(), |out| {
        let y = out.0.prev[0].as_ref().unwrap().data.get().tanh();
        out.0.prev[0].as_ref().unwrap().grad.update(|v| v + (1.0 - y*y) * out.0.grad.get());
    };
    unary relu, "ReLU" => |x| x.0.data.get().max(0.0), |out| {
        let g = if out.0.data.get() > 0.0 { out.0.grad.get() } else { 0.0 };
        out.0.prev[0].as_ref().unwrap().grad.update(|v| v + g);
    };
}

impl Value {
    pub fn from<T: Into<Value>>(t: T) -> Self {
        t.into()
    }

    fn new(value: ValueData) -> Self {
        Value(Rc::new(value))
    }

    pub fn data(&self) -> f32 {
        self.0.data.get()
    }

    pub fn grad(&self) -> f32 {
        self.0.grad.get()
    }

    pub fn zero_grad(&self) {
        self.0.grad.set(0.0);
    }

    pub fn adjust(&self, val: f32) {
        let data = &self.0.data;
        let grad = &self.0.grad;
        data.update(|v| v + val * grad.get());
    }

    pub fn sub(a: &Value, b: &Value) -> Self {
        Value::add(a, &Value::mul(b, &Value::from(-1.0)))
    }

    pub fn div(a: &Value, b: &Value) -> Self {
        Value::mul(a, &b.powneg())
    }

    pub fn pow(&self, b: &Value) -> Value {
        Value::pow_op(self, b)
    }

    pub fn matmul<const M: usize, const N: usize>(a: &[[Value; N]; M], b: &[Value; N]) -> [Value; M] {
        from_fn(|i| a[i].iter().zip(b.iter()).map(|(a, b)| a * b).reduce(|a, b| a + b).unwrap())
    }

    pub fn matadd<const N: usize, const M: usize>(a: &[[Value; N]; M], b: &[[Value; N]; M]) -> [[Value; N]; M] {
        from_fn(|i| from_fn(|j| &a[i][j] + &b[i][j]))
    }

    pub fn matmul_add<const P: usize, const N: usize>(a: &[[Value; P]; N], b: &[Value; P], c: &[Value; N]) -> [Value; N] {
        from_fn(|i| &a[i].iter().zip(b.iter()).map(|(a, b)| a * b).reduce(|a, b| a + b).unwrap() + &c[i])
    }

    pub fn activate<const I: usize>(a: [Value; I], b: &Activations) -> [Value; I] {
        match b {
            Activations::Linear => a,
            Activations::Tanh => from_fn(|i| a[i].tanh()),
            Activations::Relu => from_fn(|i| a[i].relu()),
        }
    }

    pub fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut sum = Value::from(0.0);
        loop {
            let val = iter.next();
            if val.is_none() {
                break;
            }
            sum = sum + val.unwrap();
        }
        sum
    }

    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<Value> = HashSet::new();
        self._build_topo(&mut topo, &mut visited);
        topo.reverse();

        self.0.grad.set(1.0);
        topo.iter().for_each(|v| {
            if let Some(backprop) = v._backward {
                backprop(&v);
            }
        });
    }

    fn _build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if visited.insert(self.clone()) {
            self.prev.iter().flatten().for_each(|child| {
                child._build_topo(topo, visited);
            });
            topo.push(self.clone());
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "Value(data={}, grad={})", self.data(), self.grad())
    }
}

/*
----------------------------------------------------------------------------------
Rust requires this boilerplate for stuff like hashset, derefrenceing into etc.
----------------------------------------------------------------------------------
*/
impl ops::Deref for Value {
    type Target = Rc<ValueData>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Into<f32>> From<T> for Value {
    fn from(t: T) -> Self {
        Value::new(ValueData::new(t.into(), None, [None, None], None))
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
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
            impl ::std::ops::$Trait<Value> for Value {
                type Output = Value;
                fn $method(self, other: Value) -> Self::Output {
                    Value::$method(&self, &other)
                }
            }
            impl<'a,'b> ::std::ops::$Trait<&'b Value> for &'a Value {
                type Output = Value;
                fn $method(self, other: &'b Value) -> Self::Output {
                    Value::$method(self, other)
                }
            }
        )*
    }
}
impl_ops!(Add::add, Sub::sub, Mul::mul, Div::div);

impl ::std::ops::Neg for Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        Value::mul(&self, &Value::from(-1.0))
    }
}
impl<'a> ::std::ops::Neg for &'a Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        Value::mul(self, &Value::from(-1.0))
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Value::sum(iter)
    }
}
