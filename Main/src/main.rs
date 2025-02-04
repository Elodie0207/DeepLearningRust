use LibRS::Transition;
use LibRS::td_learning::{Sarsa, QLearning};

fn main() {
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

    println!("Test SARSA:");
    let mut sarsa = Sarsa::new(3, 2);
    sarsa.train(&transitions, 1000);
    for state in 0..3 {
        for action in 0..2 {
            println!("Q({}, {}) = {:.4}", state, action, sarsa.get_q_value(state, action));
        }
    }

    println!("\nTest Q-Learning:");
    let mut q_learning = QLearning::new(3, 2);
    q_learning.train(&transitions, 1000);
    for state in 0..3 {
        for action in 0..2 {
            println!("Q({}, {}) = {:.4}", state, action, q_learning.get_q_value(state, action));
        }
    }
}