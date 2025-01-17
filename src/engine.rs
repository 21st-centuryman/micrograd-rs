use std::{
    cell::{Ref, RefCell},
    collections::HashSet,
    fmt::{Debug, Formatter, Result},
    hash::{Hash, Hasher},
    iter::Sum,
    ops,
    rc::Rc,
};

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Value(Rc<RefCell<Values>>);

pub struct Values {
    pub data: f64,
    pub grad: f64,
    pub op: Option<String>,
    pub prev: Vec<Value>,
    pub _backward: Option<fn(value: &Ref<Values>)>,
}

impl Values {
    fn new(data: f64, op: Option<String>, prev: Vec<Value>, _backward: Option<fn(value: &Ref<Values>)>) -> Values {
        Values {
            data,
            grad: 0.0,
            op,
            prev,
            _backward,
        }
    }
}

impl Value {
    pub fn from<T: Into<Value>>(t: T) -> Value {
        t.into()
    }

    fn new(value: Values) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.borrow().grad
    }

    pub fn zero_grad(&self) {
        self.borrow_mut().grad = 0.0;
    }

    pub fn adjust(&self, val: f64) {
        let mut value = self.borrow_mut();
        value.data += val * value.grad;
    }

    pub fn add(a: &Value, b: &Value) -> Value {
        let _backward: fn(value: &Ref<Values>) = |out| {
            out.prev[0].borrow_mut().grad += out.grad;
            out.prev[1].borrow_mut().grad += out.grad;
        };

        Value::new(Values::new(
            a.borrow().data + b.borrow().data,
            Some("+".to_string()),
            vec![a.clone(), b.clone()],
            Some(_backward),
        ))
    }

    pub fn mul(a: &Value, b: &Value) -> Value {
        let _backward: fn(value: &Ref<Values>) = |out| {
            out.prev[0].borrow_mut().grad += out.prev[1].borrow().data * out.grad;
            out.prev[1].borrow_mut().grad += out.prev[0].borrow().data * out.grad;
        };

        Value::new(Values::new(
            a.borrow().data * b.borrow().data,
            Some("*".to_string()),
            vec![a.clone(), b.clone()],
            Some(_backward),
        ))
    }

    pub fn pow(&self, other: &Value) -> Value {
        let _backward: fn(value: &Ref<Values>) = |out| {
            let mut base = out.prev[0].borrow_mut(); // I want to remove this
            base.grad += out.prev[1].borrow().data * (base.data.powf(out.prev[1].borrow().data - 1.0)) * out.grad;
        };

        Value::new(Values::new(
            self.borrow().data.powf(other.borrow().data),
            Some("^".to_string()),
            vec![self.clone(), other.clone()],
            Some(_backward),
        ))
    }

    // Negative power ie x^-1, this will allow us to divide
    pub fn powneg(&self) -> Value {
        let _backward: fn(value: &Ref<Values>) = |out| {
            let mut base = out.prev[0].borrow_mut(); // and remove this
            base.grad += -(1.0 / base.data.powf(2.0)) * out.grad;
        };

        Value::new(Values::new(
            1.0 / self.borrow().data,
            Some("^".to_string()),
            vec![self.clone()],
            Some(_backward),
        ))
    }
    pub fn tanh(&self) -> Value {
        let _backward: fn(value: &Ref<Values>) = |out| {
            let out1 = out.prev[0].borrow().data.tanh();
            let mut outue = out.prev[0].borrow_mut();
            outue.grad += (1.0 - out1.powf(2.0)) * out.grad;
        };

        Value::new(Values::new(
            self.borrow().data.tanh(),
            Some("tanh".to_string()),
            vec![self.clone()],
            Some(_backward),
        ))
    }

    pub fn exp(self) -> Value {
        let _backward: fn(value: &Ref<Values>) = |out| {
            out.prev[0].borrow_mut().grad += out.data * out.grad;
        };
        Value::new(Values::new(
            self.borrow().data.exp(),
            Some("exp".to_string()),
            vec![self.clone()],
            Some(_backward),
        ))
    }

    pub fn relu(self) -> Value {
        let _backward: fn(value: &Ref<Values>) = |out| {
            out.prev[0].borrow_mut().grad += (out.data > 0.0) as i8 as f64 * out.grad;
        };

        Value::new(Values::new(
            self.borrow().data.max(0.0),
            Some("ReLU".to_string()),
            vec![self.clone()],
            Some(_backward),
        ))
    }

    pub fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        // iter.fold(Value::from(0.0), |sum, val| sum + val) <- For some reason I couldn't get this
        // to work
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

        self.borrow_mut().grad = 1.0;
        for v in topo {
            if let Some(backprop) = v.borrow()._backward {
                backprop(&v.borrow());
            }
        }
    }

    fn _build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if visited.insert(self.clone()) {
            self.borrow().prev.iter().for_each(|child| {
                child._build_topo(topo, visited);
            });
            topo.push(self.clone());
        }
    }
}

impl Debug for Values {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "Value(data={}, grad={})", self.data, self.grad)
    }
}

/*
----------------------------------------------------------------------------------
Rust requires this boilerplate for stuff like hashset, derefrenceing into etc.
----------------------------------------------------------------------------------
*/
impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state);
    }
}

impl ops::Deref for Value {
    type Target = Rc<RefCell<Values>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(Values::new(t.into(), None, Vec::new(), None))
    }
}

impl PartialEq for Values {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.grad == other.grad && self.op == other.op && self.prev == other.prev
    }
}

impl Eq for Values {}

impl Hash for Values {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.grad.to_bits().hash(state);
        self.op.hash(state);
        self.prev.hash(state);
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
        Value::mul(&self, &Value::from(-1))
    }
}

impl<'a> ops::Neg for &'a Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        Value::mul(self, &Value::from(-1))
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Value::sum(iter)
    }
}
