//! Neural networks
use crate::matrix::*;

/// a network
#[derive(Debug)]
pub struct Network {
    // activation functions
    activations: Vec<Box<dyn Activation>>,
    // topology
    topology: Vec<usize>,
    // weights
    weights: Vec<Matrix>
}

impl Network {
    /// create a new random network with the given topology and activation functions
    pub fn new_random(activations: Vec<Box<dyn Activation>>, topology: Vec<usize>) -> Network {
        let l = topology.len();
        assert!(l>1);
        let rw = initial_random_weights(&topology);
        assert_eq!(rw.len(), activations.len());
        Network {
            activations,
            topology,
            weights: rw
        }
    }

    /// create a new network with the given topology and activation functions and initial weights
    pub fn new(activations: Vec<Box<dyn Activation>>, topology: Vec<usize>, weights: &[f64]) -> Network {
        let l = topology.len();
        assert!(l>1);
        
        let rw = initial_weights(&topology, weights);
        assert_eq!(rw.len(), activations.len());
        Network {
            activations,
            topology,
            weights: rw
        }
    }

    /// forward evaluation
    pub fn forward(&self, inputs :&[f64]) -> Vec<Vec<f64>> {
        assert_eq!(self.topology[0],inputs.len());
        let mut m = Matrix::new(1,inputs.len(),inputs);
        let mut all_results = Vec::with_capacity(self.topology.len() - 1);
        self.weights.iter().enumerate().for_each(| (ix,wm) | {
            add_column(&mut m,vec!(1.0));
            m = mul(&m,wm);
            //println!("after mul: {:?}",m);
            let acts = self.activations[ix].activate(&get_data(&m));
            m = Matrix::new(size(&m).0,size(&m).1,&acts);
            //println!("after activation: {:?}",m);
            all_results.push(acts);
        });
        assert_eq!(*self.topology.last().expect("empty topology!"), 
            all_results.last().expect("empty result!").len());
        all_results
    }

    /// back propagation
    pub fn backward(&mut self, inputs :&[f64], outputs :Vec<Vec<f64>>, target :&[f64], learning_rate: f64 ) {
        debug!("Error: {}", error(target, outputs.last().expect("outputs")));
        let l = outputs.len();
        let mut new_weights = self.weights.clone();
        let mut new_targets = vec!();
        for (order,weights) in self.weights.iter().rev().enumerate() {
            let rev_order = l-order-1;
            let previous_outputs =  if rev_order>0 {
                        &outputs[rev_order-1]
                    } else {
                        inputs
                    };
            let previous_size = size(&weights).0;
            debug!("previous size: {}",previous_size);
            debug!("weights to update: {:?}",size(&weights));
            new_targets.push(vec!(0.0; previous_size));
            for (i,o) in outputs[rev_order].iter().enumerate() {
                let diff = if order==0 {
                    o - target[i]
                } else {
                    new_targets[order-1][i]
                };
                let deriv = self.activations[rev_order].derive(&outputs[rev_order],i);
                let d = diff * deriv;
                debug!("Output: {} -> {}", o, d);
                for r in 0..previous_size{
                    let w = get(&weights,r,i);
                    let p = if r<previous_outputs.len() {previous_outputs[r]} else {1.0};
                    let uw = update_weight(w, d * p, learning_rate);
                    debug!("Weight for row {}: {} -> {} -> {}", r, w, uw, w*d);
                    set(&mut new_weights[rev_order],r,i,uw);
                    new_targets[order][r]+=w*d;
                }
                
            }
            debug!("New targets: {:?}",new_targets);
        }
        debug!("Before backprop: {:?}",self.weights);
        self.weights=new_weights;
    }

    /// train for one input and one target
    pub fn train(&mut self, inputs :&[f64], target :&[f64], learning_rate: f64, max_iter: usize, max_error: f64) -> (usize,f64) {
        let mut err;
        for it in 0..max_iter {
            let outputs = self.forward(inputs);
            err = error(target, outputs.last().expect("outputs"));
            if err < max_error {
                return (it,err);
            }
            self.backward(inputs,outputs,target,learning_rate);
        }
        let outputs = self.forward(inputs);
        err = error(target, outputs.last().expect("outputs"));
        (max_iter,err)
    }

    /// online training for multiple input/targets
    pub fn train_online(&mut self, tests: &[Test], learning_rate: f64, max_iter: usize) -> f64 {
        for _ in 0..max_iter {
            
            for test in tests {
                let outputs = self.forward(&test.input);
                self.backward(&test.input,outputs,&test.target,learning_rate);
            }

        }
        tests.iter().map(|t| {
            let outputs = self.forward(&t.input);
            error(&t.target, outputs.last().expect("outputs"))
        }).sum()
        
    }
}

// generate random initial weights
fn initial_random_weights(topology: &Vec<usize>) -> Vec<Matrix> {
    let mut v = Vec::new();
    topology.iter().fold(None,
        |prev, &sz| {
            if let Some(psz) = prev {
                v.push(Matrix::rand_range(psz+1,sz,0.0,1.0));
            }
            Some(sz)
        }
        
    );
    assert_eq!(topology.len()-1,v.len());
    v
}

// build a network from initial weights
fn initial_weights(topology: &Vec<usize>, weights: &[f64]) -> Vec<Matrix> {
    let mut v = Vec::new();
    let mut st = 0;
    topology.iter().fold(None,
        |prev, &sz| {
            if let Some(psz) = prev {
                let end =st + (psz+1) * sz;
                assert!(end <= weights.len());
                v.push(Matrix::new(psz+1,sz,&weights[st..end]));
                st = end;
            }
            Some(sz)
        }
        
    );
    assert_eq!(st,weights.len());
    assert_eq!(topology.len()-1,v.len());
    v
}

/// sigmoid function
pub fn sigmoid(i: f64) -> f64 {
    f64::powi(1.0 + f64::exp(-i),-1)
}

/// hyperbolic tangent
pub fn hyptan(i: f64) -> f64 {
    let minus2 = f64::exp(-2.0 * i);
    (1.0-minus2)/(1.0+minus2)
}

/// RELU function
pub fn relu(i:f64) -> f64 {
    f64::max(0.0,i)
}

/// softmax function
pub fn softmax(v: &[f64]) -> Vec<f64> {
    let mut v2 = Vec::with_capacity(v.len());
    let d = v.iter().max_by(|x,y| x.partial_cmp(y).expect("NaN")).expect("empty vector");
    let s = v.iter().fold(0.0,|s,w|{
        let e=f64::exp(*w-d);
        v2.push(e);
        s+e
    });
    if s == 0.0 {
        v2
    } else {
        v2.iter().map(|w| w/s).collect()
    }
} 

/// error: sum of errors squared
pub fn error(target: &[f64], output: &[f64]) -> f64 {
    target.iter().zip(output.iter()).map(|(t,o)| f64::powi(t-o,2)/2.0).sum()
}

/*fn diff_deriv(target: f64, output: f64) -> f64 {
    let diff = output - target;
    let deriv = output * (1.0 - output);
    diff * deriv
}

fn weight_error(target: f64, output: f64, weighted_input: f64) -> f64 {
    diff_deriv(target,output) * weighted_input
}*/

// get the updated value for a weight
fn update_weight(old: f64, error: f64, learning_rate: f64) -> f64 {
    old - error * learning_rate
}

/// an activation function
pub trait Activation : std::fmt::Debug {
    /// forward activation of all inputs
    fn activate(&self, inputs: &[f64]) -> Vec<f64>;
    /// derivation for one output given all the outputs and the output index
    fn derive(&self, outputs: &[f64], index: usize) -> f64;
}

/// Sigmoid activation function
#[derive(Debug)]
pub struct Sigmoid{}

impl Activation for Sigmoid {
    fn activate(&self, inputs: &[f64]) -> Vec<f64> {
        inputs.iter().map(|v| sigmoid(*v)).collect()
    }  
    fn derive(&self, outputs: &[f64], index: usize) -> f64 {
        outputs[index] * (1.0 - outputs[index])
    }
}

/// Relu activation function
#[derive(Debug)]
pub struct Relu{}

impl Activation for Relu {
    fn activate(&self, inputs: &[f64]) -> Vec<f64> {
        inputs.iter().map(|v| relu(*v)).collect()
    }  
    fn derive(&self, outputs: &[f64], index: usize) -> f64 {
        if outputs[index] > 0.0 {1.0} else {0.0}
    }
}

/// Softmax activation function
#[derive(Debug)]
pub struct Softmax{}

impl Activation for Softmax {
    fn activate(&self, inputs: &[f64]) -> Vec<f64> {
        softmax(inputs)
    }  
    fn derive(&self, outputs: &[f64], index: usize) -> f64 {
        let s: f64 = outputs.iter().sum();
        let el = outputs[index];
        (s-el)*el / s.powi(2)
    }
}

/// Encapsulate one possible input and the target output, for training
pub struct Test {
    pub input: Vec<f64>,
    pub target: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mattmazur() {
        // <https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/>
        let mut nn = Network::new(vec!(Box::new(Sigmoid{}),Box::new(Sigmoid{})),vec!(2,2,2),&vec!(
            0.15, 0.25, 0.2, 0.3, 0.35, 0.35,
            0.4, 0.5, 0.45, 0.55, 0.6, 0.6
        ));
        let f1s = nn.forward(&vec!(0.05,0.1));
        assert_eq!(f1s[0],vec!(0.5932699921071872, 0.596884378259767));
        assert_eq!(f1s[1],vec!(0.7513650695523157, 0.7729284653214625));
        assert_eq!(0.2983711087600027,error(&vec!(0.01,0.99), &f1s[1]));
        // assert_eq!(0.08216704056423078, weight_error(0.01, 0.7513650695523157, f1s[0][0]));
        assert_eq!(0.35891647971788465, update_weight(0.4, 0.08216704056423078, 0.5));

        nn.backward(&vec!(0.05,0.1),f1s,&vec!(0.01,0.99), 0.5);
        let f2s = nn.forward(&vec!(0.05,0.1));
        //assert_eq!(0.29102777369359933,error(&vec!(0.01,0.99), &f2s[1]));
        // we update the biases too
        assert_eq!(0.28047144679143016,error(&vec!(0.01,0.99), &f2s[1]));

        let last_error=nn.train(&vec!(0.05,0.1),&vec!(0.01,0.99),0.5,10000,0.0000000001);
        assert!(last_error.1<0.00001);
        assert!(last_error.0<=10000);
    }

    #[test]
    fn test_becominghuman(){
        // <https://becominghuman.ai/making-a-simple-neural-network-2ea1de81ec20>
        let mut nn = Network::new_random(vec!(Box::new(Sigmoid{}),Box::new(Sigmoid{})),vec!(4,4,1));
        let tests = vec!(
            Test{input:vec!(1.0,0.0,0.0,0.0),target:vec!(0.0)},
            Test{input:vec!(0.0,1.0,0.0,0.0),target:vec!(0.0)},
            Test{input:vec!(0.0,0.0,1.0,0.0),target:vec!(1.0)},
            Test{input:vec!(0.0,0.0,0.0,1.0),target:vec!(0.0)},
        );
        let last_error=nn.train_online(&tests,0.5,10000);
        println!("last: {}",last_error);
        assert!(last_error<0.001);
    }

    #[test]
    fn test_towards(){
        // <https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6>
         let mut nn = Network::new_random(vec!(Box::new(Sigmoid{}),Box::new(Sigmoid{})),vec!(3,4,1));
        let tests = vec!(
            Test{input:vec!(0.0,0.0,1.0),target:vec!(0.0)},
            Test{input:vec!(0.0,1.0,1.0),target:vec!(1.0)},
            Test{input:vec!(1.0,0.0,1.0),target:vec!(1.0)},
            Test{input:vec!(1.0,1.0,1.0),target:vec!(0.0)},
        );
        let last_error=nn.train_online(&tests,0.5,1500);
        println!("last: {}",last_error);
        assert!(last_error<0.05);
    }

    #[test]
    fn test_softmax(){
        assert_eq!(vec!(0.09003057317038046, 0.24472847105479764, 0.6652409557748218),softmax(&vec!(1.0,2.0,3.0)));
        assert_eq!(vec!(0.0, 0.0, 1.0),softmax(&vec!(1000.0,2000.0,3000.0)));

    }
}

