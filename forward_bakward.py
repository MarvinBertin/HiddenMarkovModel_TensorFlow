from __future__ import print_function
import tensorflow as tf
import numpy as np

__author__ = 'MarvinBertin'

"Inspired by Zach Dwiel's HMM implementation"

class HiddenMarkovModel_FB(object):

    """
    Hidden Markov Model Class

    Parameters:
    -----------
    
    - S: Number of states.
    - T: Transition matrix of size S by S
         stores probability from state i to state j.
    - E: Emission matrix of size S by N (number of observations)
         stores the probability of observing  O_j  from state  S_i. 
    - T0: Initial state probabilities of size S.
    """

    def __init__(self, T, E, T0):
        # Number of states
        self.S = T.shape[0]
        
        # Emission probability
        self.E = tf.constant(E, name='emission_matrix')

        # Transition matrix
        self.T = tf.constant(T, name='transition_matrix')

        # Initial state vector
        self.T0 = tf.constant(T0, name='inital_state_vector')

    def initialize_path_variables(self, shape):
        
        pathStates = tf.Variable(tf.zeros(shape, dtype=tf.int64), name='States_matrix')
        pathScores = tf.Variable(tf.zeros(shape, dtype=tf.float64), name='Score_matrix')
        states_seq = tf.Variable(tf.zeros([shape[0]], dtype=tf.int64), name='States_sequence')
        return pathStates, pathScores, states_seq
    
    def belief_propagation(self, scores):
        
        scores_reshape = tf.reshape(scores, (-1,1))
        return tf.add(scores_reshape, tf.log(self.T))
    
    def viterbi_inference(self, obs_seq):
        
        # length of observed sequence
        self.N = len(obs_seq)
        
        # shape path Variables
        shape = [self.N, self.S]
        
        # observed sequence
        x = tf.constant(obs_seq, name='observation_sequence')
        
        # Initialize variables
        pathStates, pathScores, states_seq = self.initialize_path_variables(shape)       
        
        # log probability of emission sequence
        obs_prob_seq = tf.log(tf.gather(self.E, x))
        obs_prob_list = tf.split(0, self.N, obs_prob_seq)

        # initialize with state starting log-priors
        pathScores = tf.scatter_update(pathScores, 0, tf.log(self.T0) + tf.squeeze(obs_prob_list[0]))
            
        
        for step, obs_prob in enumerate(obs_prob_list[1:]):
            
            # propagate state belief
            belief = self.belief_propagation(pathScores[step, :])

            # the inferred state by maximizing global function
            # and update state and score matrices 
            pathStates = tf.scatter_update(pathStates, step + 1, tf.argmax(belief, 0))
            pathScores = tf.scatter_update(pathScores, step + 1, tf.reduce_max(belief, 0) + tf.squeeze(obs_prob))

        # infer most likely last state
        states_seq = tf.scatter_update(states_seq, self.N-1, tf.argmax(pathScores[self.N-1, :], 0))
        
        for step in range(self.N - 1, 0, -1):
            # for every timestep retrieve inferred state
            state = states_seq[step]
            idx = tf.reshape(tf.pack([step, state]), [1, -1])
            state_prob = tf.gather_nd(pathStates, idx)
            states_seq = tf.scatter_update(states_seq, step - 1,  state_prob[0])

        return states_seq, tf.exp(pathScores) # turn scores back to probabilities
    
    def run_viterbi(self, obs_seq):
        with tf.Session() as sess:
            
            state_graph, state_prob_graph = self.viterbi_inference(obs_seq)
            sess.run(tf.initialize_all_variables())
            states_seq, state_prob = sess.run([state_graph, state_prob_graph])

        return states_seq, state_prob 
    
    
    def initialize_variables(self, shape, shape_ext):
        self.forward = tf.Variable(tf.zeros(shape_ext, dtype=tf.float64), name='forward')
        self.backward = tf.Variable(tf.zeros(shape_ext, dtype=tf.float64), name='backward')
        self.posterior = tf.Variable(tf.zeros(shape, dtype=tf.float64), name='posteriror')


    def _forward(self, obs_prob_seq):
        # initialize with state starting priors
        self.forward = tf.scatter_update(self.forward, 0, self.T0)

        # propagate belief
        for step in range(self.N):
            # previous state probability
            prev_prob = tf.reshape(self.forward[step, :], [1, -1])
            # transition prior
            prior_prob = tf.matmul(prev_prob, self.T)
            # forward belief propagation
            forward_score = tf.multiply(prior_prob, tf.cast(obs_prob_seq[step, :], tf.float64))
            # Normalize score into a probability
            forward_prob = tf.reshape(forward_score / tf.reduce_sum(forward_score), [-1])
            # Update forward matrix
            self.forward = tf.scatter_update(self.forward, step + 1, forward_prob)

        # remove initial probability
        #self.forward = tf.slice(self.forward, [1,0], [self.N, self.S]) 
        

    def _backward(self, obs_prob_seq):
        # initialize with state ending priors
        self.backward = tf.scatter_update(self.backward, self.N, tf.ones([self.S], dtype=tf.float64)) 

        for step in range(self.N, 0, -1):
            # next state probability
            next_prob = tf.reshape(self.backward[step, :], [-1, 1])
            # observation emission probabilities
            obs_prob = tf.diag(obs_prob_seq[step - 1, :])
            # transition prior
            prior_prob = tf.matmul(self.T, obs_prob)
            # backward belief propagation
            backward_score = tf.matmul(prior_prob, next_prob)
            # Normalize score into a probability
            backward_prob = tf.reshape(backward_score / tf.reduce_sum(backward_score), [-1])

            # Update backward matrix
            self.backward = tf.scatter_update(self.backward, step - 1, backward_prob)
        
        # remove final probability
        #self.backward = tf.slice(self.backward, [0,0], [self.N, self.S])

        
    def forward_backward(self, obs_seq):
        """
        runs forward backward algorithm on observation sequence

        Arguments
        ---------
        - obs_seq : matrix of size N by S, where N is number of timesteps and
            S is the number of states

        Returns
        -------
        - forward : matrix of size N by S representing
            the forward probability of each state at each time step
        - backward : matrix of size N by S representing
            the backward probability of each state at each time step
        - posterior : matrix of size N by S representing
            the posterior probability of each state at each time step
        """

        # length of observed sequence
        self.N = len(obs_seq)

        # shape of Variables
        shape = [self.N, self.S]
        shape_ext = [self.N+1, self.S]
        
        # observed sequence
        x = tf.constant(obs_seq, dtype=tf.int32, name='observation_sequence')
        
        # initialize variables
        self.initialize_variables(shape, shape_ext)
        
        # probability of emission sequence
        obs_prob_seq = tf.gather(self.E, x)
        
        # forward belief propagation
        self._forward(obs_prob_seq)
        
        # backward belief propagation
        self._backward(obs_prob_seq)

        # posterior score
        self.posterior = tf.multiply(self.forward, self.backward)
        
        # marginal per timestep
        marginal = tf.reduce_sum(self.posterior, 1)
        
        # Normalize porsterior into probabilities
        self.posterior = self.posterior / tf.reshape(marginal, [-1, 1])

        return self.forward, self.backward, self.posterior
    
    def run_forward_backward(self, obs_seq):
        with tf.Session() as sess:
            
            forward, backward, posterior = self.forward_backward(obs_seq)
            sess.run(tf.initialize_all_variables())
            return sess.run([forward, backward, posterior])