use std::{
    array::from_fn,
    cell::RefCell,
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
    pub data: RefCell<f32>,
    pub grad: RefCell<f32>,
    pub op: Option<&'static str>,
    pub prev: Vec<Value>,
    pub _backward: Option<fn(value: &Value)>,
}

#[derive(PartialEq)]
pub enum Activations {
    Linear,
    Relu,
    Softmax,
    Tanh,
}

impl ValueData {
    fn new(data: f32, op: Option<&'static str>, prev: Vec<Value>, _backward: Option<fn(value: &Value)>) -> ValueData {
        ValueData {
            data: RefCell::new(data),
            grad: RefCell::new(0.0),
            op,
            prev,
            _backward,
        }
    }
}

impl Value {
    pub fn from<T: Into<Value>>(t: T) -> Self {
        t.into()
    }

    fn new(value: ValueData) -> Self {
        Value(Rc::new(value))
    }

    pub fn data(&self) -> f32 {
        *self.0.data.borrow()
    }

    pub fn grad(&self) -> f32 {
        *self.0.grad.borrow()
    }

    pub fn zero_grad(&self) {
        *self.0.grad.borrow_mut() = 0.0;
    }

    pub fn adjust(&self, val: f32) {
        let data = &self.0.data;
        let grad = &self.0.grad;
        *data.borrow_mut() += val * *grad.borrow();
    }

    pub fn add(a: &Value, b: &Value) -> Self {
        let _backward: fn(value: &Value) = |out| {
            *out.0.prev[0].0.grad.borrow_mut() += *out.0.grad.borrow();
            *out.0.prev[1].0.grad.borrow_mut() += *out.0.grad.borrow();
        };

        Value::new(ValueData::new(
            *a.0.data.borrow() + *b.0.data.borrow(),
            Some("+"),
            vec![a.clone(), b.clone()],
            Some(_backward),
        ))
    }

    pub fn mul(a: &Value, b: &Value) -> Self {
        let _backward: fn(value: &Value) = |out| {
            let a_data = *out.prev[0].data.borrow();
            let b_data = *out.prev[1].data.borrow();
            *out.prev[0].grad.borrow_mut() += b_data * *out.0.grad.borrow();
            *out.prev[1].grad.borrow_mut() += a_data * *out.0.grad.borrow();
        };

        Value::new(ValueData::new(
            *a.data.borrow() * *b.data.borrow(),
            Some("*"),
            vec![a.clone(), b.clone()],
            Some(_backward),
        ))
    }

    pub fn matadd<const N: usize, const M: usize>(a: &[[Value; N]; M], b: &[[Value; N]; M]) -> [[Value; N]; M] {
        from_fn(|i| from_fn(|j| &a[i][j] + &b[i][j]))
    }

    pub fn matmul<const P: usize, const N: usize>(a: &[[Value; P]; N], b: &[Value; P]) -> [Value; N] {
        from_fn(|i| a[i].iter().zip(b.iter()).map(|(a, b)| a * b).reduce(|a, b| a + b).unwrap())
    }

    pub fn matmul_add<const P: usize, const N: usize>(a: &[[Value; P]; N], b: &[Value; P], c: &[Value; N]) -> [Value; N] {
        from_fn(|i| &a[i].iter().zip(b.iter()).map(|(a, b)| a * b).reduce(|a, b| a + b).unwrap() + &c[i])
    }

    pub fn activate<const I: usize>(a: [Value; I], b: &Activations) -> [Value; I] {
        match b {
            Activations::Linear => a,
            Activations::Tanh => from_fn(|i| a[i].tanh()),
            Activations::Relu => from_fn(|i| a[i].relu()),
            Activations::Softmax => Value::softmax(&a),
        }
    }

    pub fn pow(&self, other: &Value) -> Value {
        let _backward: fn(value: &Value) = |out| {
            let exponent_value = *out.prev[1].data.borrow();
            *out.prev[0].grad.borrow_mut() += exponent_value * (*out.prev[0].data.borrow()).powf(exponent_value - 1.0) * *out.grad.borrow();
        };

        Value::new(ValueData::new(
            self.data.borrow().powf(*other.data.borrow()),
            Some("^"),
            vec![self.clone(), other.clone()],
            Some(_backward),
        ))
    }

    // Negative power ie x^-1, this will allow us to divide
    pub fn powneg(&self) -> Value {
        let _backward: fn(value: &Value) = |out| {
            let base = &out.0.prev[0];
            let mut grad = base.0.grad.borrow_mut();
            *grad += -(1.0 / (*base.0.data.borrow()).powf(2.0)) * *out.0.grad.borrow();
        };

        Value::new(ValueData::new(1.0 / *self.data.borrow(), Some("^"), vec![self.clone()], Some(_backward)))
    }

    pub fn exp(&self) -> Value {
        let _backward: fn(value: &Value) = |out| {
            *out.prev[0].grad.borrow_mut() += *out.0.data.borrow() * *out.0.grad.borrow();
        };
        Value::new(ValueData::new(
            self.0.data.borrow().exp(),
            Some("exp"),
            vec![self.clone()],
            Some(_backward),
        ))
    }

    pub fn ln(&self) -> Value {
        let _backward: fn(value: &Value) = |out| {
            let x = *out.prev[0].data.borrow();
            *out.prev[0].grad.borrow_mut() += *out.0.grad.borrow() / x;
        };
        Value::new(ValueData::new(
            self.0.data.borrow().ln(),
            Some("log"),
            vec![self.clone()],
            Some(_backward),
        ))
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

        *self.0.grad.borrow_mut() = 1.0;
        topo.iter().for_each(|v| {
            if let Some(backprop) = v._backward {
                backprop(&v);
            }
        });
    }

    fn _build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if visited.insert(self.clone()) {
            self.prev.iter().for_each(|child| {
                child._build_topo(topo, visited);
            });
            topo.push(self.clone());
        }
    }

    /*
    ----------------------------------------------------------------------------------
    Activation functions
    ----------------------------------------------------------------------------------
    */

    pub fn tanh(&self) -> Value {
        let _backward: fn(value: &Value) = |out| {
            let out1 = out.prev[0].0.data.borrow().tanh();
            let mut outue = out.prev[0].grad.borrow_mut();
            *outue += (1.0 - out1.powf(2.0)) * *out.0.grad.borrow();
        };

        Value::new(ValueData::new(
            self.0.data.borrow().tanh(),
            Some("tanh"),
            vec![self.clone()],
            Some(_backward),
        ))
    }

    pub fn relu(&self) -> Value {
        let _backward: fn(&Value) = |out| {
            let cond = *out.0.data.borrow() > 0.0;
            let grad = if cond { *out.0.grad.borrow() } else { 0.0 };
            *out.prev[0].grad.borrow_mut() += grad;
        };

        Value::new(ValueData::new(
            self.0.data.borrow().max(0.0),
            Some("ReLU"),
            vec![self.clone()],
            Some(_backward),
        ))
    }

    pub fn softmax<const I: usize>(a: &[Value; I]) -> [Value; I] {
        let _backward: fn(value: &Value) = |out| {
            let probs: Vec<f32> = out
                .prev
                .iter()
                .map(|v| (*v.data.borrow() - f32::MAX).exp() / out.prev.iter().map(|v| (*v.data.borrow() - f32::MAX).exp()).sum::<f32>())
                .collect();

            let y_i = *out.0.data.borrow();
            let g_i = *out.0.grad.borrow();

            for (j, logit) in out.prev.iter().enumerate() {
                let y_j = probs[j];
                let partial = if (y_j - y_i).abs() < 1e-12 { y_i * (1.0 - y_j) } else { -y_i * y_j };
                *logit.grad.borrow_mut() += partial * g_i;
            }
        };

        let summed: f32 = a.iter().map(|x| (x.data() - f32::MAX).exp()).sum();
        from_fn(|i| {
            let val = (a[i].data() - f32::MAX).exp() / summed;
            Value::new(ValueData::new(val, Some("softmax"), a.to_vec(), Some(_backward)))
        })
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
        Value::new(ValueData::new(t.into(), None, Vec::new(), None))
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
This allows us to use Value + Value instead of Value.add(Value), just so it works like micrograd
------------------------------------------------------------------------------------------------
*/
impl ops::Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        Value::add(&self, &other)
    }
}

impl<'a, 'b> ops::Add<&'b Value> for &'a Value {
    type Output = Value;
    fn add(self, other: &'b Value) -> Self::Output {
        Value::add(self, other)
    }
}

impl ops::Sub<Value> for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        Value::add(&self, &(-other))
    }
}

impl<'a, 'b> ops::Sub<&'b Value> for &'a Value {
    type Output = Value;

    fn sub(self, other: &'b Value) -> Self::Output {
        Value::add(self, &(-other))
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        Value::mul(&self, &other)
    }
}

impl<'a, 'b> ops::Mul<&'b Value> for &'a Value {
    type Output = Value;

    fn mul(self, other: &'b Value) -> Self::Output {
        Value::mul(self, other)
    }
}

impl ops::Div<Value> for Value {
    type Output = Value;

    fn div(self, other: Value) -> Self::Output {
        Value::mul(&self, &other.powneg())
    }
}

impl<'a, 'b> ops::Div<&'b Value> for &'a Value {
    type Output = Value;

    fn div(self, other: &'b Value) -> Self::Output {
        Value::mul(self, &other.powneg())
    }
}

impl ops::Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        Value::mul(&self, &Value::from(-1.0))
    }
}

impl<'a> ops::Neg for &'a Value {
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
