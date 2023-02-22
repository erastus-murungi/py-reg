use std::hash::Hash;

use crate::matching::{Context, Cursor};

#[derive(Debug)]
pub struct ParsingResult {}

#[derive(Debug)]
pub struct ParsingState<'a> {
    group_count: u8,
    position: usize,
    regex: &'a Vec<char>,
}

pub trait Node<'a> {
    type T;

    fn to_string(&self) -> &'a str;

    fn accept(&self) -> Self::T;
}

pub trait Matcher: Hash {
    fn accepts(&self, cursor: Cursor, context: Context) -> bool;
}

#[derive(PartialEq, PartialOrd, Debug, Hash)]
pub struct Character {
    c: char,
}

impl Character {
    pub fn new(c: char) -> Character {
        Character { c: c }
    }
}

impl Matcher for Character {
    fn accepts(&self, cursor: Cursor, context: Context) -> bool {
        return self.c == context.text[cursor.position];
    }
}

#[derive(Debug)]
pub struct Parser<'a> {
    parsing_state: ParsingState<'a>,
}

impl<'a> Parser<'a> {
    pub fn new(regex: &Vec<char>) -> Parser {
        Parser {
            parsing_state: ParsingState {
                group_count: 0,
                position: 0,
                regex: regex,
            },
        }
    }
    fn parse(&self) -> Result<ParsingResult, &'static str> {
        return Err("not implemented");
    }

    fn consume(&mut self, c: char) -> Result<char, &'static str> {
        let position = self.parsing_state.position;
        let regex = self.parsing_state.regex;

        if position >= regex.len() {
            return Err("index out of bounds error");
        } else if regex[position] != c {
            return Err("character mismatch");
        } else {
            self.parsing_state.position += 1;
            return Ok(c);
        }
    }
}
