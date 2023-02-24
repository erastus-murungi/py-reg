use std::{
    collections::{HashMap, HashSet},
    env::temp_dir,
    fmt::format,
    fs::File,
    io::{self, Write},
    process::Command,
};

use itertools::Itertools;

use crate::{
    parser::{run_parse, visitor::Visitor, Data, Node, ParserError, Quantifier},
    utils::RegexFlags,
};

type State = usize;

#[derive(Hash, Debug)]
struct Transition {
    node: Node,
    end: State,
}

impl Transition {
    pub fn new(matcher: Node, end: State) -> Self {
        Self { node: matcher, end }
    }
}
#[derive(Debug)]
pub struct RegexNFA {
    state_counter: usize,
    pattern: String,
    flags: RegexFlags,
    start: State,
    alphabet: HashSet<Node>,
    transitions: HashMap<State, Vec<Transition>>,
    accept: State,
    states: HashSet<State>,
}

#[derive(Debug)]
pub enum ReError {
    ParsingFailed(ParserError),
    CompilationError,
}

type Fragment = (State, State);

impl RegexNFA {
    pub fn new(pattern: &str) -> RegexNFA {
        RegexNFA {
            state_counter: Default::default(),
            pattern: String::from(pattern),
            flags: RegexFlags::OPTIMIZE,
            start: Default::default(),
            alphabet: HashSet::new(),
            transitions: HashMap::new(),
            accept: Default::default(),
            states: HashSet::new(),
        }
    }

    pub fn gen_state(&mut self) -> State {
        self.state_counter += 1;
        self.state_counter
    }

    pub fn fragment(&mut self) -> Fragment {
        (self.gen_state(), self.gen_state())
    }

    pub fn compile(&mut self) -> Result<(), ReError> {
        match run_parse(&self.pattern, &mut self.flags) {
            Ok(root) => {
                let (start, accept) = root.accept(self);
                self.start = start;
                self.accept = accept;
                Ok(())
            }
            Err(parsing_error) => Err(ReError::ParsingFailed(parsing_error)),
        }
    }

    pub fn add_transition(&mut self, start: State, end: State, matcher: Node) -> () {
        match self.transitions.get_mut(&start) {
            Some(transitions) => transitions.push(Transition::new(matcher, end)),
            None => {
                let mut transitions = Vec::new();
                transitions.push(Transition::new(matcher, end));
                self.transitions.insert(start, transitions);
            }
        }
    }

    pub fn epsilon(&mut self, start: State, end: State) -> () {
        self.add_transition(start, end, Node::Epsilon)
    }

    fn symbol_transition(&mut self, node: Node) -> (State, State) {
        let (start, end) = self.fragment();
        self.add_transition(start, end, node);
        return (start, end);
    }

    fn alternation(&mut self, lower: &Fragment, upper: &Fragment) -> Fragment {
        let fragment = self.fragment();
        self.epsilon(fragment.0, lower.0);
        self.epsilon(fragment.0, upper.0);
        self.epsilon(lower.1, fragment.1);
        self.epsilon(upper.1, fragment.1);

        fragment
    }

    /// Convert the automata to a GraphViz Dot code for the deubgging purposes.
    pub fn as_graphviz_code(&self) -> Result<(), io::Error> {
        let mut out = String::new();
        let mut seen: HashSet<State> = HashSet::new();
        for (start, transitions) in self.transitions.iter() {
            for transition in transitions {
                let opts = if *start == self.start {
                    ""
                } else {
                    "[fillcolor=\"#EEEEEE\" fontcolor=\"#888888\"]"
                };
                if !seen.contains(start) {
                    out += &format!("node_{}[label=\"{}\"]{}\n", start, start, opts);
                    seen.insert(*start);
                }
                if !seen.contains(&transition.end) {
                    if transition.end == self.accept {
                        out += &format!(
                            "node_{}[label=\"{}\"shape=doublecircle]{}\n",
                            transition.end, transition.end, opts
                        );
                    } else {
                        out += &format!(
                            "node_{}[label=\"{}\"]{}\n",
                            transition.end, transition.end, opts
                        );
                    }

                    seen.insert(transition.end);
                }
                if let Node::Epsilon = transition.node {
                    out += &format!("node_{} -> node_{}[style=dashed]\n", start, transition.end);
                } else {
                    out += &format!(
                        "node_{} -> node_{}[label=\"{}\"]\n",
                        start, transition.end, transition.node
                    );
                }
            }
        }
        let opts = "node [shape=circle style=filled fillcolor=\"#4385f5\" fontcolor=\"#FFFFFF\" \
        color=white penwidth=5.0 margin=0.1 width=0.5 height=0.5 fixedsize=true]";
        let graph_dot = format!(
            "digraph G {{  rankdir=\"LR\" graph [fontname = \"Courier New\"];
                node [fontname = \"verdana\", style = rounded];
                edge [fontname = \"verdana\"];
                {{\n{}\n{}\n}}}}",
            opts, out
        );

        println!("{}", graph_dot);

        let mut dir = temp_dir();
        dir.push("fsm.dot");

        let mut file = File::create(dir.clone())?;
        if let Ok(_) = file.write_all(graph_dot.as_bytes()) {
            let dot_exec_filepath = if cfg!(target_os = "macos") {
                "/usr/local/bin/dot"
            } else {
                "/usr/bin/dot"
            };
            let dot_filepath = format!("{}", dir.to_str().unwrap());

            let mut output_dir = temp_dir();
            output_dir.push("graph.pdf");
            let output_filepath = format!("{}", output_dir.to_str().unwrap());
            let mut output = Command::new(dot_exec_filepath)
                .args(["-Tpdf", "-Gdpi=96", &dot_filepath, "-o", &output_filepath])
                .spawn()
                .expect("Failed to execute command");
            println!("{:#?}", output.wait());
            output = Command::new("open")
                .arg(&output_filepath)
                .spawn()
                .expect("Failed to execute command");
            println!("{:#?}", output.wait());
            output = Command::new("rm")
                .arg(output_filepath)
                .spawn()
                .expect("Failed to execute command");
            println!("{:#?}", output.stdout);
        }

        Ok(())
    }
}

impl Visitor for RegexNFA {
    type Result = (State, State);

    fn visit_expression(&mut self, expression: Node) -> Self::Result {
        if let Node::Expression(items, alternate_expression) = expression {
            let fragments: Vec<Self::Result> = items.iter().map(|node| node.accept(self)).collect();
            for ((_, a_end), (b_start, _)) in fragments.iter().tuple_windows() {
                self.epsilon(*a_end, *b_start);
            }
            let fragment = (fragments.first().unwrap().0, fragments.last().unwrap().1);
            if let Some(alternative) = alternate_expression {
                let alt_fragment = alternative.accept(self);
                return self.alternation(&fragment, &alt_fragment);
            } else {
                return fragment;
            }
        } else {
            panic!("expected expression")
        }
    }

    fn visit_character(&mut self, char: Node) -> Self::Result {
        self.symbol_transition(char)
    }

    fn visit_anchor(&mut self, anchor: Node) -> Self::Result {
        self.symbol_transition(anchor)
    }

    fn visit_dot(&mut self, dot: Node) -> Self::Result {
        self.symbol_transition(dot)
    }

    fn visit_match(&mut self, match_: Node) -> Self::Result {
        if let Node::Match(item, quantifier) = match_ {
            if let Quantifier::None = quantifier {
                item.accept(self)
            } else {
                todo!()
            }
        } else {
            panic!()
        }
    }

    fn visit_character_group(&mut self, character_group: Node) -> Self::Result {
        self.symbol_transition(character_group)
    }

    fn visit_group(&mut self, group: Node) -> Self::Result {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visitor_creation() {
        let pattern = String::from("a[0-9]\\w");
        let mut regex = RegexNFA::new(&pattern);
        println!("{:#?}", regex.compile());
        println!("{:#?}", regex);
        regex.as_graphviz_code();
    }
}
