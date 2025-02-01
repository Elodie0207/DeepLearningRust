use std::f64;
use rand::Rng;

const GAMMA: f64 = 0.9; // Facteur de discount
const THETA: f64 = 0.001; // Seuil de convergence
const ALPHA: f64 = 0.1; // Taux d'apprentissage pour TD Learning
const EPSILON: f64 = 0.1; // Pour la politique ε-greedy

struct Transition {
    next: usize,   // État suivant
    reward: f64,   // Récompense pour cette transition
    proba: f64,    // Probabilité de cette transition
}

// Structure pour stocker la Q-table
struct QTable {
    values: Vec<Vec<f64>>, // Q(s,a) pour chaque paire état-action
}

impl QTable {
    fn new(n_states: usize, n_actions: usize) -> Self {
        QTable {
            values: vec![vec![0.0; n_actions]; n_states],
        }
    }

    fn get(&self, state: usize, action: usize) -> f64 {
        self.values[state][action]
    }

    fn set(&mut self, state: usize, action: usize, value: f64) {
        self.values[state][action] = value;
    }

    // Sélectionner une action selon la politique ε-greedy
    fn select_action(&self, state: usize, n_actions: usize) -> usize {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < EPSILON {
            // Exploration: action aléatoire
            rng.gen_range(0..n_actions)
        } else {
            // Exploitation: meilleure action
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

// Implémentation de SARSA
struct Sarsa {
    q_table: QTable,
}

impl Sarsa {
    fn new(n_states: usize, n_actions: usize) -> Self {
        Sarsa {
            q_table: QTable::new(n_states, n_actions),
        }
    }

    fn update(&mut self, state: usize, action: usize, reward: f64,
              next_state: usize, next_action: usize) {
        let current_q = self.q_table.get(state, action);
        let next_q = self.q_table.get(next_state, next_action);

        // Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        let new_q = current_q + ALPHA * (reward + GAMMA * next_q - current_q);

        self.q_table.set(state, action, new_q);
    }

    fn select_action(&self, state: usize, n_actions: usize) -> usize {
        self.q_table.select_action(state, n_actions)
    }
}

// Implémentation de Q-Learning
struct QLearning {
    q_table: QTable,
}

impl QLearning {
    fn new(n_states: usize, n_actions: usize) -> Self {
        QLearning {
            q_table: QTable::new(n_states, n_actions),
        }
    }

    fn update(&mut self, state: usize, action: usize, reward: f64, next_state: usize) {
        let current_q = self.q_table.get(state, action);

        // Trouver la meilleure action suivante
        let max_next_q = (0..self.q_table.values[0].len())
            .map(|a| self.q_table.get(next_state, a))
            .fold(f64::NEG_INFINITY, f64::max);

        // Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        let new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q);

        self.q_table.set(state, action, new_q);
    }

    fn select_action(&self, state: usize, n_actions: usize) -> usize {
        self.q_table.select_action(state, n_actions)
    }
}

fn main() {
    let states = vec![0, 1, 2];
    let actions = vec![0, 1];
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

    // Test avec la Policy Iteration classique
    println!("Policy Iteration Results:");
    let value_function = iterative_policy_evaluation(&states, &actions, &transitions);
    for (state, value) in value_function.iter().enumerate() {
        println!("V({}) = {:.4}", state, value);
    }

    // Test avec SARSA
    println!("\nSARSA Learning:");
    let mut sarsa = Sarsa::new(states.len(), actions.len());

    // Simulation de 1000 épisodes pour SARSA
    for episode in 0..1000 {
        let mut state = 0; // État initial
        let mut action = sarsa.select_action(state, actions.len());

        // Un épisode continue jusqu'à l'état terminal ou max steps
        for _ in 0..100 {
            // Obtenir la transition pour cette action
            let transition = &transitions[state][action][0];
            let next_state = transition.next;
            let reward = transition.reward;

            let next_action = sarsa.select_action(next_state, actions.len());
            sarsa.update(state, action, reward, next_state, next_action);

            state = next_state;
            action = next_action;

            if state == 2 { // État terminal
                break;
            }
        }
    }

    // Afficher les Q-values finales pour SARSA
    println!("Final Q-values for SARSA:");
    for state in 0..states.len() {
        for action in 0..actions.len() {
            println!("Q({}, {}) = {:.4}", state, action, sarsa.q_table.get(state, action));
        }
    }

    // Test avec Q-Learning
    println!("\nQ-Learning:");
    let mut q_learning = QLearning::new(states.len(), actions.len());

    // Simulation de 1000 épisodes pour Q-Learning
    for episode in 0..1000 {
        let mut state = 0; // État initial

        // Un épisode continue jusqu'à l'état terminal ou max steps
        for _ in 0..100 {
            let action = q_learning.select_action(state, actions.len());

            // Obtenir la transition pour cette action
            let transition = &transitions[state][action][0];
            let next_state = transition.next;
            let reward = transition.reward;

            q_learning.update(state, action, reward, next_state);

            state = next_state;

            if state == 2 { // État terminal
                break;
            }
        }
    }

    // Afficher les Q-values finales pour Q-Learning
    println!("Final Q-values for Q-Learning:");
    for state in 0..states.len() {
        for action in 0..actions.len() {
            println!("Q({}, {}) = {:.4}", state, action, q_learning.q_table.get(state, action));
        }
    }
}

fn iterative_policy_evaluation(
    states: &[usize],
    actions: &[usize],
    transitions: &[Vec<Vec<Transition>>],
) -> Vec<f64> {
    let mut value_function = vec![0.0; states.len()];

    loop {
        let mut delta: f64 = 0.0;

        for &state in states {
            let mut value = 0.0;
            let prob = 1.0 / actions.len() as f64;

            for &action in actions {
                for transition in &transitions[state][action] {
                    value += prob * transition.proba
                        * (transition.reward + GAMMA * value_function[transition.next]);
                }
            }

            delta = delta.max((value - value_function[state]).abs());
            value_function[state] = value;
        }

        if delta < THETA {
            break;
        }
    }

    value_function
}