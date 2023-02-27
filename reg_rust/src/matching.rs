use std::{
    collections::{HashSet, VecDeque},
    iter::FusedIterator,
};

use crate::{fsm::RegexNFA, fsm::Transition, parser::Node, utils::RegexFlags};

#[derive(Debug)]
pub struct Cursor {
    pub position: usize,
    pub groups: Vec<Option<usize>>,
}

impl Cursor {
    pub fn new(position: usize, n_groups: usize) -> Cursor {
        return Cursor {
            position: position,
            groups: vec![None; n_groups * 2],
        };
    }

    pub fn update(&self, node: Node) -> Cursor {
        match node {
            Node::GroupEntry(index) => {
                let mut copy = self.groups.clone();
                copy[index * 2] = Some(self.position);
                Cursor {
                    position: self.position,
                    groups: copy,
                }
            }
            Node::GroupExit(index) => {
                let mut copy = self.groups.clone();
                copy[index * 2 + 1] = Some(self.position);
                Cursor {
                    position: self.position,
                    groups: copy,
                }
            }
            _ => Cursor {
                position: self.position + node.increment(),
                groups: self.groups.clone(),
            },
        }
    }

    fn to_string(&self, text: &str) -> Vec<Option<String>> {
        (0..self.groups.len() / 2)
            .map(|group_index| {
                let (some_frm, some_to) = (
                    self.groups[group_index * 2],
                    self.groups[group_index * 2 + 1],
                );
                if let Some(frm) = some_frm {
                    if let Some(to) = some_to {
                        return Some(text[frm..to].to_string());
                    }
                }
                return None;
            })
            .collect()
    }
}

#[derive(Debug, Hash)]
pub struct Context {
    pub text: Vec<char>,
    pub flags: RegexFlags,
}

impl<'a> Context {
    pub fn new(text: Vec<char>) -> Context {
        return Context {
            text: text,
            flags: RegexFlags::NO_FLAG,
        };
    }

    pub fn new_with_flags(text: Vec<char>, flags: RegexFlags) -> Context {
        return Context {
            text: text,
            flags: flags,
        };
    }
}

#[derive(Debug)]
pub struct Match {
    start: usize,
    end: usize,
    text: String,
    captured_groups: Vec<Option<String>>,
}

impl Match {
    pub fn new(
        start: usize,
        end: usize,
        text: String,
        captured_groups: Vec<Option<String>>,
    ) -> Self {
        Match {
            start,
            end,
            text,
            captured_groups,
        }
    }

    pub fn span(&self) -> (usize, usize) {
        (self.start, self.end)
    }

    pub fn groups(&self) -> Vec<Option<String>> {
        self.captured_groups.clone()
    }

    fn group(&self, index: usize) -> Option<String> {
        if index > self.captured_groups.len() {
            panic!("group index out of bounds");
        }
        if index == 0 {
            Some(self.text[self.start..self.end].to_string())
        } else {
            self.captured_groups[index].clone()
        }
    }
}

pub(crate) trait Matcher<'a>
where
    Self: Sized,
{
    fn group_count(&self) -> usize;
    fn get_flags(&self) -> RegexFlags;
    fn match_suffix(&self, cursor: Cursor, context: &'a Context) -> Option<Cursor>;
    fn is_match(&self, text: &'a str) -> bool;
    fn find(&self, text: &'a str) -> Option<String>;
    fn find_iter(&'a self, text: &'a str) -> Box<dyn Iterator<Item = Match> + 'a>;
}

#[derive(Debug)]
struct RegexNFAMatches<'a> {
    text: &'a str,
    nfa: &'a RegexNFA,
    start: usize,
    context: Context,
    increment: usize,
}

impl<'a> Iterator for RegexNFAMatches<'a> {
    type Item = Match;

    fn next(&mut self) -> Option<Match> {
        while self.start <= self.text.len() {
            if let Some(cursor) = self.nfa.match_suffix(
                Cursor::new(self.start, self.nfa.group_count()),
                &self.context,
            ) {
                self.increment = if cursor.position == self.start {
                    1
                } else {
                    cursor.position - self.start
                };
                let match_result = Some(Match::new(
                    self.start,
                    cursor.position,
                    self.text[self.start..cursor.position].to_string(),
                    cursor.to_string(self.text),
                ));
                self.start += self.increment;
                return match_result;
            }
            self.start += 1;
        }
        return None;
    }
}

impl<'a> FusedIterator for RegexNFAMatches<'a> {}

impl<'a> Matcher<'a> for RegexNFA {
    fn get_flags(&self) -> RegexFlags {
        self.get_flags()
    }

    fn find(&self, text: &'a str) -> Option<String> {
        self.find_iter(text).next().map(|m| m.group(0)).unwrap()
    }

    fn match_suffix(&self, cursor: Cursor, context: &'a Context) -> Option<Cursor> {
        let mut visited: HashSet<(usize, &Transition)> = HashSet::new();
        let mut queue = VecDeque::from(self.step(
            &Transition::new(Node::Epsilon, self.start),
            &cursor,
            context,
            &mut visited,
        ));

        let mut match_result: Option<Cursor> = None;
        loop {
            let mut frontier: VecDeque<(Transition, Cursor)> = VecDeque::new();
            visited = HashSet::new();

            while let Some((transition, cursor)) = queue.pop_front() {
                if transition.node.accepts(&cursor, context) {
                    if self.accept == transition.end {
                        match_result = Some(cursor.update(transition.node));
                        break;
                    }
                    frontier.extend(self.step(&transition, &cursor, context, &mut visited));
                } else if let Node::Epsilon = transition.node {
                    if self.accept == transition.end {
                        match_result = Some(cursor.update(transition.node));
                        break;
                    }
                    frontier.extend(self.step(&transition, &cursor, context, &mut visited));
                }
            }

            if frontier.is_empty() {
                break;
            }
            std::mem::swap(&mut frontier, &mut queue);
        }
        match_result
    }

    fn is_match(&self, text: &'a str) -> bool {
        match self.find(text) {
            Some(_) => true,
            _ => false,
        }
    }

    fn group_count(&self) -> usize {
        return self.group_count();
    }

    fn find_iter(&'a self, text: &'a str) -> Box<dyn Iterator<Item = Match> + '_> {
        Box::new(RegexNFAMatches {
            text: text,
            nfa: self,
            start: 0,
            increment: 1,
            context: Context::new_with_flags(text.chars().collect(), self.get_flags()),
        })
    }
}

#[allow(unused_imports)]
mod tests {
    use crate::{fsm::RegexNFA, matching::Matcher};

    #[test]
    fn test_simple_kleene_star() {
        let pattern = "[a-z]*";
        let regex = RegexNFA::new(&pattern);
        // regex.render();
        let items = &[
            Some("abc".to_string()),
            Some("".to_string()),
            Some("".to_string()),
        ];
        for (index, s) in regex.find_iter("abcE").map(|m| m.group(0)).enumerate() {
            assert_eq!(s, items[index]);
        }
    }

    #[test]
    fn test_sim() {
        let pattern = r"[abcd]+(x)+";
        let mut regex = RegexNFA::new(pattern);
        regex.compile().unwrap();

        for item in regex.find_iter("xxxabcdxxx") {
            println!("{:#?}", item.groups());
        }
    }
}
