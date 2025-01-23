use std::f64;

const GAMMA: f64 = 0.9; // Facteur de discount
const THETA: f64 = 0.001; // Seuil de convergence

struct Transition {
    next: usize,   // État suivant
    reward: f64,   // Récompense pour cette transition
    proba: f64,    // Probabilité de cette transition
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


    let value_function = iterative_policy_evaluation(&states, &actions, &transitions);


    for (state, value) in value_function.iter().enumerate() {
        println!("V({}) = {:.4}", state, value);
    }
}

fn iterative_policy_evaluation(
    states: &[usize],                     // Ensemble des états
    actions: &[usize],                    // Ensemble des actions
    transitions: &[Vec<Vec<Transition>>], // Modèle de transition
) -> Vec<f64> {
    let mut value_function = vec![0.0; states.len()];

    loop {
        let mut delta: f64 = 0.0;

        for &state in states {
            let mut value = 0.0;
            let prob = 1.0 / actions.len() as f64;

            for &action in actions {

                for transition in &transitions[state][action] {
                    value += prob
                        * transition.proba
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
