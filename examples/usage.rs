use micrograd::engine::Value;

fn main() {
    let a = Value::from(-4.0);
    let b = Value::from(2.0);

    let mut c = &a + &b;
    let mut d = &a * &b + b.pow(&Value::from(3.0));

    c = &Value::from(2.0) * &c + Value::from(1.0);
    c = Value::from(1.0) + &Value::from(2.0) * &c + (-&a);
    d = &d + &(&d * &Value::from(2.0)) + (&b + &a).relu();
    d = &d + &(&Value::from(3.0) * &d) + (&b - &a).relu();

    let e = &c - &d;
    let f = e.pow(&Value::from(2.0));
    let mut g = &f / &Value::from(2.0);
    g = g + &Value::from(10.0) / &f;

    println!("{:.4}", g.data.borrow()); // prints 24.7041, the outcome of this forward pass
    g.backward();
    println!("{:.4}", a.grad.borrow()); // print 138.8338, i.e. the numerical value of dg/da
    println!("{:.4}", b.grad.borrow()); // print 645.5773, i.e. the numerical value of dg/db

    println!("a is {:?}", a);
    println!("b is {:?}", b);
    println!("c is {:?}", c);
    println!("d is {:?}", d);
    println!("e is {:?}", e);
    println!("f is {:?}", f);
    println!("g is {:?}", g);
}
