pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

// ajout des alogorithme Sarsa et Episodic Semi Gradient SARSA

use std::f64;
use rand::Rng;

pub const GAMMA: f64 = 0.9;
pub const THETA: f64 = 0.001;
pub const ALPHA: f64 = 0.1;
pub const EPSILON: f64 = 0.1;

pub struct Transition {
    pub next: usize,
    pub reward: f64,
    pub proba: f64,
}

pub struct QTable {
    values: Vec<Vec<f64>>,
}

impl QTable {
    pub fn new(n_states: usize, n_actions: usize) -> Self {
        QTable {
            values: vec![vec![0.0; n_actions]; n_states],
        }
    }

    pub fn get(&self, state: usize, action: usize) -> f64 {
        self.values[state][action]
    }

    pub fn set(&mut self, state: usize, action: usize, value: f64) {
        self.values[state][action] = value;
    }

    pub fn select_action(&self, state: usize, n_actions: usize) -> usize {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < EPSILON {
            rng.gen_range(0..n_actions)
        } else {
            let mut best_action = 0;
            let mut best_value = f64::NEG_INFINITY;

            for action in 0..n_actions {
                let value = self.get(state, action);
                if value > best_value {
                    best_value = value;
                    best_action = action;
                }
            }
            best_action
        }
    }
}

// Module TD Learning
pub mod td_learning {
    use super::*;

    // SARSA
    pub struct Sarsa {
        q_table: QTable,
    }

    impl Sarsa {
        pub fn new(n_states: usize, n_actions: usize) -> Self {
            Sarsa {
                q_table: QTable::new(n_states, n_actions),
            }
        }

        pub fn update(&mut self, state: usize, action: usize, reward: f64,
                      next_state: usize, next_action: usize) {
            let current_q = self.q_table.get(state, action);
            let next_q = self.q_table.get(next_state, next_action);
            let new_q = current_q + ALPHA * (reward + GAMMA * next_q - current_q);
            self.q_table.set(state, action, new_q);
        }

        pub fn select_action(&self, state: usize, n_actions: usize) -> usize {
            self.q_table.select_action(state, n_actions)
        }

        pub fn train(&mut self, transitions: &[Vec<Vec<Transition>>], episodes: usize) {
            for _ in 0..episodes {
                let mut state = 0;
                let mut action = self.select_action(state, transitions[0].len());

                for _ in 0..100 {
                    let transition = &transitions[state][action][0];
                    let next_state = transition.next;
                    let reward = transition.reward;

                    let next_action = self.select_action(next_state, transitions[0].len());
                    self.update(state, action, reward, next_state, next_action);

                    state = next_state;
                    action = next_action;

                    if state == transitions.len() - 1 {
                        break;
                    }
                }
            }
        }

        pub fn get_q_value(&self, state: usize, action: usize) -> f64 {
            self.q_table.get(state, action)
        }
    }

    // Q-Learning
    pub struct QLearning {
        q_table: QTable,
    }

    impl QLearning {
        pub fn new(n_states: usize, n_actions: usize) -> Self {
            QLearning {
                q_table: QTable::new(n_states, n_actions),
            }
        }

        pub fn update(&mut self, state: usize, action: usize, reward: f64, next_state: usize) {
            let current_q = self.q_table.get(state, action);
            let max_next_q = (0..self.q_table.values[0].len())
                .map(|a| self.q_table.get(next_state, a))
                .fold(f64::NEG_INFINITY, f64::max);

            let new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q);
            self.q_table.set(state, action, new_q);
        }

        pub fn select_action(&self, state: usize, n_actions: usize) -> usize {
            self.q_table.select_action(state, n_actions)
        }

        pub fn train(&mut self, transitions: &[Vec<Vec<Transition>>], episodes: usize) {
            for _ in 0..episodes {
                let mut state = 0;

                for _ in 0..100 {
                    let action = self.select_action(state, transitions[0].len());

                    let transition = &transitions[state][action][0];
                    let next_state = transition.next;
                    let reward = transition.reward;

                    self.update(state, action, reward, next_state);
                    state = next_state;

                    if state == transitions.len() - 1 {
                        break;
                    }
                }
            }
        }

        pub fn get_q_value(&self, state: usize, action: usize) -> f64 {
            self.q_table.get(state, action)
        }
    }
}

#[cfg(test)]
mod tests1 {
    use super::*;
    use super::td_learning::*;

    #[test]
    fn test_td_learning() {
        let transitions = vec![
            vec![
                vec![Transition { next: 0, reward: 0.0, proba: 1.0 }],
                vec![Transition { next: 1, reward: 1.0, proba: 1.0 }],
            ],
            vec![
                vec![Transition { next: 2, reward: 2.0, proba: 1.0 }],
                vec![Transition { next: 0, reward: 0.0, proba: 1.0 }],
            ],
            vec![
                vec![Transition { next: 2, reward: 0.0, proba: 1.0 }],
                vec![Transition { next: 2, reward: 0.0, proba: 1.0 }],
            ],
        ];

        // Test SARSA
        let mut sarsa = Sarsa::new(3, 2);
        sarsa.train(&transitions, 1000);

        // Test Q-Learning
        let mut q_learning = QLearning::new(3, 2);
        q_learning.train(&transitions, 1000);

        // Vérifie que les Q-values ne sont pas toutes nulles
        assert!(sarsa.get_q_value(0, 0) != 0.0 || sarsa.get_q_value(0, 1) != 0.0);
        assert!(q_learning.get_q_value(0, 0) != 0.0 || q_learning.get_q_value(0, 1) != 0.0);
    }
}