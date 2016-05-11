# Hidden Markov Model in TensorFlow

## Viterbi Algorithm
* Efficient way of finding the most likely state sequence.
* Method is general statistical framework of compound decision theory. 
* Maximizes a posteriori probability recursively.
* Assumed to have a finite-state discrete-time Markov process.

## Forward-Backward Algorithm
* The goal of the forward-backward algorithm is to find the conditional distribution over hidden states given the data.
* It is used to find the most likely state for any point in time.
* It cannot, however, be used to find the most likely sequence of states (see Viterbi)

## Baum Welch Algorithm
  * Expectation Maximization Inference of unknown parameters of a Hidden Markov Model.

# Viterbi

![](images/viterbi.png)

**Unrolled graph**

![](images/viterbi2.png)

**Belief Propagation**

![](images/viterbi3.png)

**Backtrack**

![](images/viterbi4.png)

# Baum Welch and Forward-Backward

![](images/BW5.png)

**Forward-Backward**

![](images/BW6.png)

**Re-estimate**

![](images/BW4.png)

**Unrolled**

![](images/BW3.png)
