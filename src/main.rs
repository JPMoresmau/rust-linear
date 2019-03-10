use linear::nn::{Network,Sigmoid};
extern crate env_logger;

// Run some custom code to exercise the library
fn main() {
    env_logger::init();
    /*let nn = linear::nn::Network::new(linear::nn::sigmoid,vec!(2,3,1));
    println!("initial: {:?}",nn);
    let f1 = nn.forward(&vec!(1.0,0.0));
    println!("initial: {:?}",f1);
    let f2 = nn.forward(&vec!(0.0,0.0));
    println!("initial: {:?}",f2);*/
    let mut nn = Network::new(
        vec!(Box::new(Sigmoid{}),Box::new(Sigmoid{})),
        vec!(2,2,2),
        &vec!(
            0.15, 0.25, 0.2, 0.3, 0.35, 0.35,
            0.4, 0.5, 0.45, 0.55, 0.6, 0.6
    ));
    println!("initial: {:?}",nn);
    /*let f1 = nn.forward(&vec!(0.05,0.1));
    
    //println!("first result: {:?}",f1);
    println!("Error: {}", linear::nn::error(&vec!(0.01,0.99), f1.last().unwrap()));
    nn.backward(&vec!(0.05,0.1),f1,&vec!(0.01,0.99), 0.5);
    let f2 = nn.forward(&vec!(0.05,0.1));
    println!("Error: {}", linear::nn::error(&vec!(0.01,0.99), f2.last().unwrap()));
    //println!("second result: {:?}",f2);
    println!("final: {:?}",nn.weights);
    nn.backward(&vec!(0.05,0.1),f2,&vec!(0.01,0.99), 0.5);*/
    let last_error=nn.train(&vec!(0.05,0.1),&vec!(0.01,0.99),0.5,10000,0.0000000001);
    println!("last error: {:?}",last_error);
    println!("final: {:?}",nn);
}
