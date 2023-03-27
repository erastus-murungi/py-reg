use std::{
    collections::{HashMap, HashSet, VecDeque},
    hash::Hash,
};

use itertools::Itertools;

use crate::{
    fsm::ReError,
    matching::Cursor,
    parser::{run_parse, visitor::Visitor, Quantifier, UpperBound},
    parser::{Data, Node},
    utils::RegexFlags,
};

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum Instruction {
    End,
    EmptyString,
    Jump(Box<Instruction>),
    Fork(Box<Instruction>, Box<Instruction>),
    Consume(Box<Node>),
    Capture(Box<Node>),
}

#[derive(Debug, Clone)]
pub struct PikeVM {
    pub next: HashMap<Instruction, Instruction>,
    pub root: Instruction,
    pub group_count: usize,
    pub flags: RegexFlags,
}

pub type Thread<'inst> = (Instruction, Cursor);
pub type VMSearchSpaceNode<'inst> = (usize, Instruction);

impl PikeVM {
    fn add_capturing_markers(
        &mut self,
        codes: &(Instruction, Instruction),
        group_index: Option<usize>,
    ) -> (Instruction, Instruction) {
        match group_index {
            Some(index) => {
                let capturing_anchors = (
                    Instruction::Capture(Box::new(Node::GroupEntry(index))),
                    Instruction::Capture(Box::new(Node::GroupExit(index))),
                );
                self.next
                    .insert(capturing_anchors.0.clone(), codes.0.clone());
                self.next
                    .insert(codes.1.clone(), capturing_anchors.1.clone());
                capturing_anchors
            }
            _ => (codes.0.clone(), codes.1.clone()),
        }
    }

    fn zero_or_one(
        &mut self,
        codes: &(Instruction, Instruction),
        lazy: bool,
    ) -> (Instruction, Instruction) {
        let empty = Instruction::EmptyString;
        self.next.insert(codes.1.clone(), empty.clone());
        (
            if lazy {
                Instruction::Fork(Box::new(empty.clone()), Box::new(codes.0.clone()))
            } else {
                Instruction::Fork(Box::new(codes.0.clone()), Box::new(empty.clone()))
            },
            empty,
        )
    }

    fn one_or_more(
        &mut self,
        codes: &(Instruction, Instruction),
        lazy: bool,
    ) -> (Instruction, Instruction) {
        let empty = Instruction::EmptyString;
        let fork = if lazy {
            Instruction::Fork(Box::new(empty.clone()), Box::new(codes.0.clone()))
        } else {
            Instruction::Fork(Box::new(codes.0.clone()), Box::new(empty.clone()))
        };
        self.next.insert(codes.1.clone(), empty.clone());
        (fork, empty)
    }

    fn zero_or_more(
        &mut self,
        codes: &(Instruction, Instruction),
        lazy: bool,
    ) -> (Instruction, Instruction) {
        let empty = Instruction::EmptyString;
        let fork = if lazy {
            Instruction::Fork(Box::new(empty.clone()), Box::new(codes.0.clone()))
        } else {
            Instruction::Fork(Box::new(codes.0.clone()), Box::new(empty.clone()))
        };
        self.next
            .insert(codes.1.clone(), Instruction::Jump(Box::new(empty.clone())));
        (fork, empty)
    }

    fn apply_range_quantifier(
        &mut self,
        node: Node,
        lower: u64,
        upperbound: UpperBound,
        lazy: bool,
    ) -> (Instruction, Instruction) {
        if lower == 0 {
            if let UpperBound::Unbounded = upperbound {
                let codes = node.accept(self);
                return self.zero_or_more(&codes, lazy);
            }
            if let UpperBound::Undefined = upperbound {
                return (Instruction::EmptyString, Instruction::EmptyString);
            }
        }
        let mut fragments: Vec<(Instruction, Instruction)> = Vec::new();
        match upperbound {
            UpperBound::Unbounded => {
                for _ in 0..(lower - 1) {
                    let codes = node.accept(self);
                    fragments.push(codes);
                }
                let instructions = node.accept(self);
                fragments.push(self.one_or_more(&instructions, lazy));
            }
            UpperBound::Undefined => {
                for _ in 0..lower {
                    let frag = node.accept(self);
                    fragments.push(frag);
                }
            }
            UpperBound::Bounded(upper) => {
                for _ in 0..upper {
                    fragments.push(node.accept(self));
                }
                let empty = Instruction::EmptyString;
                for _ in lower as usize..upper as usize {
                    let codes = node.accept(self);
                    let jump = Instruction::Jump(Box::new(empty.clone()));
                    let fork = if lazy {
                        Instruction::Fork(Box::new(jump), Box::new(codes.0.clone()))
                    } else {
                        Instruction::Fork(Box::new(codes.0.clone()), Box::new(jump))
                    };
                    fragments.push((fork, codes.1));
                }
                fragments.push((empty.clone(), empty))
            }
        }
        for (a, b) in fragments.iter().tuple_windows() {
            self.next.insert(a.1.clone(), b.0.clone());
        }
        (
            fragments.first().unwrap().0.clone(),
            fragments.last().unwrap().1.clone(),
        )
    }

    fn match_or_group(&mut self, node: Node) -> (Instruction, Instruction) {
        match node {
            Node::Group(node, group_index, quantifier) => {
                let instructions = node.accept(self);
                match quantifier {
                    Quantifier::None => self.add_capturing_markers(&instructions, group_index),
                    Quantifier::ZeroOrOne(lazy) => {
                        let instructions = self.add_capturing_markers(&instructions, group_index);
                        self.zero_or_one(&instructions, lazy);
                        instructions
                    }
                    Quantifier::OneOrMore(lazy) => {
                        let instructions = self.add_capturing_markers(&instructions, group_index);
                        self.one_or_more(&instructions, lazy)
                    }
                    Quantifier::ZeroOrMore(lazy) => {
                        let instructions = self.add_capturing_markers(&instructions, group_index);
                        self.zero_or_more(&instructions, lazy);
                        instructions
                    }
                    Quantifier::Range(lower, upper, lazy) => {
                        self.apply_range_quantifier(*node, lower, upper, lazy)
                    }
                }
            }
            Node::Match(node, quantifier) => match quantifier {
                Quantifier::None => node.accept(self),
                Quantifier::OneOrMore(lazy) => {
                    let instructions = node.accept(self);
                    self.one_or_more(&instructions, lazy)
                }
                Quantifier::ZeroOrOne(lazy) => {
                    let instructions = node.accept(self);
                    self.zero_or_one(&instructions, lazy);
                    instructions
                }
                Quantifier::ZeroOrMore(lazy) => {
                    let instructions = node.accept(self);
                    self.zero_or_more(&instructions, lazy)
                }
                Quantifier::Range(lower, upper, lazy) => {
                    self.apply_range_quantifier(*node, lower, upper, lazy)
                }
            },
            _ => panic!("expected Group or Match, not {:#?}", node),
        }
    }

    pub fn new(input: &str) -> Result<PikeVM, ReError> {
        let mut flags = RegexFlags::OPTIMIZE;
        PikeVM::new_with_flags(input, &mut flags)
    }
    pub fn new_with_flags(input: &str, flags: &mut RegexFlags) -> Result<PikeVM, ReError> {
        let parsing_result = run_parse(input, flags);
        match parsing_result {
            Ok((node, gc)) => {
                let mut vm = PikeVM {
                    next: HashMap::new(),
                    root: Instruction::End,
                    group_count: gc,
                    flags: *flags,
                };
                let res = node.accept(&mut vm);
                vm.root = res.0;
                vm.next.insert(res.1, Instruction::End);
                Ok(vm)
            }
            Err(err) => Err(ReError::ParsingFailed(err)),
        }
    }

    fn primitive(node: Node) -> (Instruction, Instruction) {
        (
            Instruction::Consume(Box::new(node.clone())),
            Instruction::Consume(Box::new(node)),
        )
    }

    pub fn queue_thread(
        &self,
        queue: &mut VecDeque<Thread>,
        thread: Thread,
        visited: &mut HashSet<VMSearchSpaceNode>,
    ) {
        let mut stack = vec![thread];

        while !stack.is_empty() {
            let (instruction, cursor) = stack.pop().unwrap();
            let vm_node = (cursor.position, instruction.clone());

            if visited.contains(&vm_node) {
                continue;
            }
            visited.insert(vm_node);

            match instruction.clone() {
                Instruction::EmptyString => {
                    stack.push((self.next.get(&instruction).unwrap().clone(), cursor))
                }
                Instruction::Jump(target) => stack.push((*target, cursor)),
                Instruction::Fork(preferred, alternative) => {
                    stack.push((*alternative, cursor.clone()));
                    stack.push((*preferred, cursor));
                }
                Instruction::Capture(capturing_anchor) => stack.push((
                    self.next.get(&instruction).unwrap().clone(),
                    cursor.update(&*capturing_anchor),
                )),
                _ => queue.push_front((instruction.clone(), cursor.clone())),
            }
        }
    }
}

impl Visitor for PikeVM {
    type Result = (Instruction, Instruction);

    fn visit_expression(&mut self, expression: Node) -> Self::Result {
        match expression {
            Node::Expression(items, alternate) => {
                let mut preferred_codes: Vec<(Instruction, Instruction)> = Vec::new();
                for subexpr in items {
                    preferred_codes.push(subexpr.accept(self))
                }
                for (a, b) in preferred_codes.iter().tuple_windows() {
                    self.next.insert(a.1.clone(), b.0.clone());
                }
                let codes = (
                    preferred_codes.first().unwrap().0.clone(),
                    preferred_codes.last().unwrap().1.clone(),
                );

                if let Some(node) = alternate {
                    let alt_codes = node.accept(self);
                    let empty = Instruction::EmptyString;
                    self.next
                        .insert(codes.1, Instruction::Jump(Box::new(empty.clone())));
                    self.next.insert(alt_codes.1, empty.clone());

                    return (
                        Instruction::Fork(Box::new(codes.0), Box::new(alt_codes.0)),
                        empty,
                    );
                }
                codes
            }
            _ => panic!("expected an expression type!"),
        }
    }

    fn visit_character(&mut self, character: Node) -> Self::Result {
        Self::primitive(character)
    }

    fn visit_anchor(&mut self, anchor: Node) -> Self::Result {
        match anchor {
            Node::EmptyString => (Instruction::EmptyString, Instruction::EmptyString),
            _ => Self::primitive(anchor),
        }
    }

    fn visit_dot(&mut self, dot: Node) -> Self::Result {
        Self::primitive(dot)
    }

    fn visit_match(&mut self, match_: Node) -> Self::Result {
        self.match_or_group(match_)
    }

    fn visit_character_group(&mut self, character_group: Node) -> Self::Result {
        Self::primitive(character_group)
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
        let regex = PikeVM::new(&pattern).unwrap();
        println!("{:#?}", regex);
    }
}
