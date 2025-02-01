use std::f64;
use std::collections::HashMap;

// Constants
const GAMMA: f64 = 0.9;
const THETA: f64 = 0.001;
const EPSILON: f64 = 0.1;

// Types
type State = usize;
type Action = usize;
type QFunction = HashMap<(State, Action), f64>;
type Returns = HashMap<(State, Action), Vec<f64>>;
type StateActions = Vec<(State, Action)>;
type TransitionMap = Vec<Vec<Vec<Transition>>>;

// Structures
struct Transition {
    next: usize,
    reward: f64,
    proba: f64,
}

struct Episode {
    states: Vec<State>,
    actions: Vec<Action>,
    rewards: Vec<f64>,
}

// Policy Structure
struct Policy {
    state_policies: HashMap<State, HashMap<Action, f64>>,
    states: Vec<State>,
    actions: Vec<Action>,
}

impl Policy {
    fn new(states: &[State], actions: &[Action]) -> Self {
        let mut policy = Policy {
            state_policies: HashMap::new(),
            states: states.to_vec(),
            actions: actions.to_vec(),
        };
        for &s in states {
            policy.state_policies.insert(s, HashMap::new());
            policy.update_state_policy(s, &HashMap::new());
        }
        policy
    }

    fn new_epsilon_soft(states: &[State], actions: &[Action], epsilon: f64) -> Self {
        let mut policy = Self::new(states, actions);
        for &s in states {
            for &a in actions {
                policy.state_policies.get_mut(&s).unwrap()
                    .insert(a, epsilon / actions.len() as f64);
            }
        }
        policy
    }

    fn uniform(states: &[State], actions: &[Action]) -> Self {
        let mut policy = Self::new(states, actions);
        let prob = 1.0 / actions.len() as f64;
        for &s in states {
            for &a in actions {
                policy.state_policies.get_mut(&s).unwrap().insert(a, prob);
            }
        }
        policy
    }

    fn get_greedy_action(&self, state: State) -> Action {
        self.state_policies[&state]
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&a, _)| a)
            .unwrap()
    }

    fn update_state_policy(&mut self, state: State, q_function: &QFunction) {
        let mut best_action = self.actions[0];
        let mut best_value = f64::NEG_INFINITY;

        for &action in &self.actions {
            let value = q_function.get(&(state, action)).unwrap_or(&0.0);
            if *value > best_value {
                best_value = *value;
                best_action = action;
            }
        }

        let state_policy = self.state_policies.get_mut(&state).unwrap();
        for &action in &self.actions {
            state_policy.insert(action, if action == best_action { 1.0 } else { 0.0 });
        }
    }

    fn update_epsilon_soft_policy(&mut self, state: State, best_action: Action, epsilon: f64) {
        let state_policy = self.state_policies.get_mut(&state).unwrap();
        for &action in &self.actions {
            let prob = if action == best_action {
                1.0 - epsilon + (epsilon / self.actions.len() as f64)
            } else {
                epsilon / self.actions.len() as f64
            };
            state_policy.insert(action, prob);
        }
    }
}


fn iterative_policy_evaluation(
    states: &[State],
    actions: &[Action],
    transitions: &TransitionMap,
) -> Vec<f64> {
    let mut value_function = vec![0.0; states.len()];

    loop {
        let mut delta = 0.0;
        for &state in states {
            let mut value = 0.0;
            let prob = 1.0 / actions.len() as f64;

            for &action in actions {
                for transition in &transitions[state][action] {
                    value += prob * transition.proba *
                        (transition.reward + GAMMA * value_function[transition.next]);
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

fn value_iteration(
    states: &[State],
    actions: &[Action],
    transitions: &TransitionMap,
) -> (Vec<f64>, HashMap<State, Action>) {
    let mut v = vec![0.0; states.len()];
    let mut policy: HashMap<State, Action> = HashMap::new();

    loop {
        let mut delta = 0.0;

        for &s in states {
            let old_v = v[s];
            let mut best_value = f64::NEG_INFINITY;
            let mut best_action = 0;

            for &a in actions {
                let mut new_value = 0.0;
                for transition in &transitions[s][a] {
                    new_value += transition.proba *
                        (transition.reward + GAMMA * v[transition.next]);
                }

                if new_value > best_value {
                    best_value = new_value;
                    best_action = a;
                }
            }

            v[s] = best_value;
            policy.insert(s, best_action);
            delta = delta.max((old_v - best_value).abs());
        }

        if delta < THETA {
            break;
        }
    }

    (v, policy)
}

fn policy_iteration(
    states: &[State],
    actions: &[Action],
    transitions: &TransitionMap,
) -> (Vec<f64>, HashMap<State, Action>) {
    let mut v = vec![0.0; states.len()];
    let mut policy: HashMap<State, Action> = states.iter().map(|&s| (s, 0)).collect();

    loop {
        // Policy Evaluation
        loop {
            let mut delta = 0.0;
            for &s in states {
                let tmp = v[s];
                let a = policy[&s];
                let mut new_value = 0.0;
                for transition in &transitions[s][a] {
                    new_value += transition.proba *
                        (transition.reward + GAMMA * v[transition.next]);
                }
                v[s] = new_value;
                delta = delta.max((tmp - new_value).abs());
            }
            if delta < THETA {
                break;
            }
        }

        // Policy Improvement
        let mut policy_stable = true;
        for &s in states {
            let old_action = policy[&s];
            let mut best_action = old_action;
            let mut best_value = f64::NEG_INFINITY;

            for &a in actions {
                let mut new_value = 0.0;
                for transition in &transitions[s][a] {
                    new_value += transition.proba *
                        (transition.reward + GAMMA * v[transition.next]);
                }
                if new_value > best_value {
                    best_value = new_value;
                    best_action = a;
                }
            }

            policy.insert(s, best_action);
            if old_action != best_action {
                policy_stable = false;
            }
        }

        if policy_stable {
            break;
        }
    }

    (v, policy)
}

// Monte Carlo Algorithms
fn off_policy_mc_control(
    states: &[State],
    actions: &[Action],
    generate_episode: impl Fn(&Policy) -> Episode,
) -> (QFunction, Policy) {
    let mut q_function: QFunction = HashMap::new();
    let mut c: QFunction = HashMap::new();
    let mut policy = Policy::new(states, actions);

    for &s in states {
        for &a in actions {
            q_function.insert((s, a), 0.0);
            c.insert((s, a), 0.0);
        }
    }

    loop {
        let behavior_policy = Policy::uniform(states, actions);
        let episode = generate_episode(&behavior_policy);

        let mut g = 0.0;
        let mut w = 1.0;

        for t in (0..episode.states.len()).rev() {
            let state = episode.states[t];
            let action = episode.actions[t];
            let reward = episode.rewards[t];

            g = GAMMA * g + reward;
            *c.get_mut(&(state, action)).unwrap() += w;

            let old_q = q_function[&(state, action)];
            q_function.insert(
                (state, action),
                old_q + w / c[&(state, action)] * (g - old_q)
            );

            policy.update_state_policy(state, &q_function);

            if action != policy.get_greedy_action(state) {
                break;
            }

            w *= actions.len() as f64;
        }
    }
}

fn on_policy_mc_control(
    states: &[State],
    actions: &[Action],
    generate_episode: impl Fn(&Policy) -> Episode,
    epsilon: f64,
) -> (QFunction, Policy) {
    let mut q_function: QFunction = HashMap::new();
    let mut returns: HashMap<(State, Action), Vec<f64>> = HashMap::new();
    let mut policy = Policy::new_epsilon_soft(states, actions, epsilon);

    for &s in states {
        for &a in actions {
            q_function.insert((s, a), 0.0);
            returns.insert((s, a), Vec::new());
        }
    }

    loop {
        let episode = generate_episode(&policy);
        let mut g = 0.0;
        let mut seen_pairs = StateActions::new();

        for t in (0..episode.states.len()).rev() {
            let state = episode.states[t];
            let action = episode.actions[t];
            let reward = episode.rewards[t];

            g = GAMMA * g + reward;

            if !seen_pairs.contains(&(state, action)) {
                seen_pairs.push((state, action));
                returns.get_mut(&(state, action)).unwrap().push(g);

                let state_returns = returns.get(&(state, action)).unwrap();
                q_function.insert(
                    (state, action),
                    state_returns.iter().sum::<f64>() / state_returns.len() as f64
                );

                let best_action = policy.get_greedy_action(state);
                policy.update_epsilon_soft_policy(state, best_action, epsilon);
            }
        }
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

    let (value_function, policy) = value_iteration(&states, &actions, &transitions);

    for (state, value) in value_function.iter().enumerate() {
        println!("V({}) = {:.4}", state, value);
    }

    for (state, action) in policy.iter() {
        println!("π({}) = {}", state, action);
    }
}