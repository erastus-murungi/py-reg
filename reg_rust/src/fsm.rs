use std::{
    collections::{HashMap, HashSet},
    env::temp_dir,
    fs::File,
    io::{self, Write},
    process::Command,
};

use itertools::Itertools;

use crate::{
    matching::{Context, Cursor},
    parser::{run_parse, visitor::Visitor, Data, Node, ParserError, Quantifier, UpperBound},
    utils::RegexFlags,
};

type State = usize;

#[derive(Hash, Debug, PartialEq, Eq, Clone)]
pub struct Transition {
    pub node: Node,
    pub end: State,
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
    pub start: State,
    alphabet: HashSet<Node>,
    transitions: HashMap<State, Vec<Transition>>,
    pub accept: State,
    states: HashSet<State>,
    group_count: usize,
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
            group_count: Default::default(),
        }
    }

    pub fn get_flags(&self) -> RegexFlags {
        self.flags
    }

    pub fn group_count(&self) -> usize {
        self.group_count
    }

    pub fn gen_state(&mut self) -> State {
        self.state_counter += 1;
        self.states.insert(self.state_counter);
        self.state_counter
    }

    pub fn fragment(&mut self) -> Fragment {
        (self.gen_state(), self.gen_state())
    }

    pub fn compile(&mut self) -> Result<(), ReError> {
        match run_parse(&self.pattern, &mut self.flags) {
            Ok((root, group_count)) => {
                let (start, accept) = root.accept(self);
                self.start = start;
                self.accept = accept;
                self.group_count = group_count;
                Ok(())
            }
            Err(parsing_error) => Err(ReError::ParsingFailed(parsing_error)),
        }
    }

    pub fn add_transition(&mut self, start: State, end: State, matcher: Node) -> () {
        match matcher {
            Node::GroupLink | Node::Epsilon => false,
            _ => self.alphabet.insert(matcher.clone()),
        };

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

    fn symbol_transition(&mut self, node: Node) -> Fragment {
        let (start, end) = self.fragment();
        self.add_transition(start, end, node);
        return (start, end);
    }

    fn symbol_transition_existing_fragment(
        &mut self,
        node: Node,
        fragment: Fragment,
    ) -> (State, State) {
        let (start, end) = fragment;
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

    fn zero_or_one(&mut self, fragment: &Fragment, lazy: bool) {
        self.add_transition(fragment.0, fragment.1, Node::EmptyString);
        if lazy {
            self.transitions.get_mut(&fragment.0).unwrap().reverse();
        }
    }

    fn one_or_more(&mut self, fragment: &Fragment, lazy: bool) -> Fragment {
        let s = self.gen_state();
        self.epsilon(fragment.1, fragment.0);
        self.epsilon(fragment.1, s);
        if lazy {
            self.transitions.get_mut(&fragment.1).unwrap().reverse();
        }
        (fragment.0, s)
    }

    fn zero_or_more(&mut self, fragment: &Fragment, lazy: bool) -> Fragment {
        let empty = self.symbol_transition(Node::EmptyString);

        self.epsilon(fragment.1, empty.1);
        self.epsilon(fragment.1, fragment.0);
        self.epsilon(empty.0, fragment.0);

        if !lazy {
            self.transitions.get_mut(&empty.0).unwrap().reverse();
            self.transitions.get_mut(&fragment.1).unwrap().reverse();
        }

        empty
    }

    fn add_capturing_markers(
        &mut self,
        fragment: Fragment,
        group_index: Option<usize>,
    ) -> Fragment {
        match group_index {
            Some(index) => {
                let markers = self.fragment();
                self.symbol_transition_existing_fragment(
                    Node::GroupEntry(index),
                    (markers.0, fragment.0),
                );
                self.symbol_transition_existing_fragment(
                    Node::GroupExit(index),
                    (fragment.1, markers.1),
                );

                markers
            }
            None => fragment,
        }
    }

    fn apply_range_quantifier(
        &mut self,
        node: Node,
        lower: u64,
        upperbound: UpperBound,
        lazy: bool,
    ) -> Fragment {
        if lower == 0 {
            if let UpperBound::Unbounded = upperbound {
                let frag = self.match_or_group(node);
                return self.zero_or_more(&frag, lazy);
            }
            if let UpperBound::Undefined = upperbound {
                return self.symbol_transition(Node::EmptyString);
            }
        }
        let mut fragments: Vec<Fragment> = Vec::new();
        match upperbound {
            UpperBound::Unbounded => {
                for _ in 0..(lower - 1) {
                    let frag = self.match_or_group(node.clone());
                    fragments.push(frag);
                }
                let frag = self.match_or_group(node);
                fragments.push(self.one_or_more(&frag, lazy));
            }
            UpperBound::Undefined => {
                for _ in 0..lower {
                    let frag = self.match_or_group(node.clone());
                    fragments.push(frag);
                }
            }
            UpperBound::Bounded(upper) => {
                for _ in 0..upper {
                    let frag = self.match_or_group(node.clone());
                    fragments.push(frag);
                }
                for fragment in fragments.drain(lower as usize..upper as usize) {
                    self.add_transition(fragment.0, fragment.1, Node::EmptyString);
                    if lazy {
                        self.transitions.get_mut(&fragment.0).unwrap().reverse();
                    }
                }
            }
        }
        for (a, b) in fragments.iter().tuple_windows() {
            self.epsilon(a.1, b.0);
        }
        (fragments.first().unwrap().0, fragments.last().unwrap().1)
    }

    fn match_or_group(&mut self, node: Node) -> Fragment {
        match node {
            Node::Group(node, group_index, quantifier) => {
                let fragment = node.accept(self);
                match quantifier {
                    Quantifier::None => self.add_capturing_markers(fragment, group_index),
                    Quantifier::ZeroOrOne(lazy) => {
                        let frag = self.add_capturing_markers(fragment, group_index);
                        self.zero_or_one(&frag, lazy);
                        frag
                    }
                    Quantifier::OneOrMore(lazy) => {
                        let frag = self.add_capturing_markers(fragment, group_index);
                        self.one_or_more(&frag, lazy)
                    }
                    Quantifier::ZeroOrMore(lazy) => {
                        let frag = self.add_capturing_markers(fragment, group_index);
                        self.zero_or_more(&fragment, lazy);
                        frag
                    }
                    Quantifier::Range(lower, upper, lazy) => {
                        self.apply_range_quantifier(*node, lower, upper, lazy)
                    }
                }
            }
            Node::Match(node, quantifier) => {
                let fragment = node.accept(self);
                match quantifier {
                    Quantifier::None => fragment,
                    Quantifier::OneOrMore(lazy) => self.one_or_more(&fragment, lazy),
                    Quantifier::ZeroOrOne(lazy) => {
                        self.zero_or_one(&fragment, lazy);
                        fragment
                    }
                    Quantifier::ZeroOrMore(lazy) => self.zero_or_more(&fragment, lazy),
                    Quantifier::Range(lower, upper, lazy) => {
                        self.apply_range_quantifier(*node, lower, upper, lazy)
                    }
                }
            }
            _ => panic!("expected Group or Match, not {:#?}", node),
        }
    }

    pub fn step<'c>(
        &'c self,
        start: &Transition,
        cursor: &Cursor,
        context: &Context,
        visited: &mut HashSet<(usize, &'c Transition)>,
    ) -> Vec<(Transition, Cursor)> {
        match self.transitions.get(&start.end) {
            Some(initial_transitions) => {
                let mut stack: Vec<(&Transition, Cursor)> = initial_transitions
                    .iter()
                    .map(|nxt| (nxt, cursor.update(start.node.clone())))
                    .rev()
                    .collect();
                let mut transitions: Vec<(Transition, Cursor)> = Vec::new();
                while let Some((transition, cursor)) = stack.pop() {
                    if !visited.contains(&(cursor.position, transition)) {
                        visited.insert((cursor.position, transition));

                        if transition.node.increment() == 0 {
                            if transition.node.accepts(&cursor, context) {
                                transitions.push((transition.clone(), cursor))
                            } else {
                                match transition.node {
                                    Node::Epsilon if transition.end == self.accept => {
                                        transitions.push((transition.clone(), cursor))
                                    }
                                    _ => match self.transitions.get(&transition.end) {
                                        Some(some_transitions) => stack.extend(
                                            some_transitions
                                                .iter()
                                                .map(|nxt| {
                                                    (nxt, cursor.update(transition.node.clone()))
                                                })
                                                .rev(),
                                        ),
                                        _ => {}
                                    },
                                }
                            }
                        } else {
                            transitions.push((transition.clone(), cursor))
                        }
                    }
                }
                transitions
            }
            None => Vec::new(),
        }
    }

    /// Convert the automata to a GraphViz Dot code for the deubgging purposes.
    pub fn render(&self) -> Result<(), io::Error> {
        let mut out = String::new();
        let mut seen: HashSet<State> = HashSet::new();
        for (start, transitions) in self.transitions.iter() {
            let opts = "[fillcolor=\"#EEEEEE\" fontcolor=\"#888888\"]";
            if !seen.contains(start) {
                if *start == self.start {
                    out += &format!(
                        "node_{}[label=\"{}\"]{}\n",
                        start, start, "[fillcolor=green]"
                    );
                } else {
                    out += &format!("node_{}[label=\"{}\"]{}\n", start, start, opts);
                }
                seen.insert(*start);
            }
            for transition in transitions {
                if !seen.contains(&transition.end) {
                    if transition.end == self.accept {
                        out += &format!(
                            "node_{}[label=\"{}\"shape=doublecircle]{}\n",
                            transition.end, transition.end, ""
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
        }

        Ok(())
    }
}

impl<'a> Visitor for RegexNFA {
    type Result = (State, State);

    fn visit_expression(&mut self, expression: Node) -> Self::Result {
        if let Node::Expression(items, alternate_expression) = expression {
            let fragments: Vec<Self::Result> = items.iter().map(|node| node.accept(self)).collect();
            for (a, b) in fragments.iter().tuple_windows() {
                self.epsilon(a.1, b.0);
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
        self.match_or_group(match_)
    }

    fn visit_character_group(&mut self, character_group: Node) -> Self::Result {
        self.symbol_transition(character_group)
    }

    fn visit_group(&mut self, group: Node) -> Self::Result {
        self.match_or_group(group)
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
    }
}
