use colored::Colorize;
use itertools::Itertools;

use crate::{
    matching::{Context, Cursor},
    utils::RegexFlags,
};
use core::panic;
use std::{error::Error, fmt::Display, hash::Hash, num::ParseIntError};

use self::{parser::Parser, visitor::Visitor};

mod parser {
    // we take a parsing state and return either a valid node or an error

    use std::str::Chars;

    use itertools::{peek_nth, PeekNth};

    use super::ParserError;

    static ESCAPED: &'static [char] = &[
        '$', '(', ')', '*', '+', '-', '.', '<', '=', '>', '?', '[', '\\', ']', '^', '{', '|', '}',
    ];

    static CHARACTER_CLASSES: &'static [char] = &['w', 'W', 's', 'S', 'd', 'D'];
    static ANCHORS: &'static [char] = &['A', 'z', 'Z', 'G', 'b', 'B'];

    #[derive(Debug)]
    pub struct Parser<'a> {
        regex: &'a str,
        group_count: usize,
        regex_iter: PeekNth<Chars<'a>>,
        consumed: usize,
    }

    impl<'a> PartialEq for Parser<'a> {
        fn eq(&self, other: &Self) -> bool {
            self.group_count == other.group_count
                && self.regex_iter.clone().collect::<String>()
                    == other.regex_iter.clone().collect::<String>()
        }
    }

    impl<'a> Parser<'a> {
        pub fn new(input: &'a str) -> Parser {
            Parser {
                regex: input,
                group_count: 0,
                regex_iter: peek_nth(input.chars()),
                consumed: 0,
            }
        }

        pub fn peek(&mut self) -> Result<char, ParserError> {
            match self.regex_iter.peek() {
                Some(c) => Ok(*c),
                None => Err(ParserError::UnexexpectedEOF),
            }
        }

        pub fn group_count(&self) -> usize {
            return self.group_count;
        }

        pub fn increment_group_count(&mut self) {
            self.group_count += 1;
        }

        pub fn consume(&mut self, expected: char) -> Result<char, ParserError> {
            match self.regex_iter.peek() {
                Some(actual) => {
                    if *actual == expected {
                        self.advance_by(1);
                        Ok(expected)
                    } else {
                        Err(ParserError::UnexpectedToken(self.get_remainder(), expected))
                    }
                }
                None => Err(ParserError::UnexexpectedEOF),
            }
        }

        pub fn advance_by(&mut self, by: usize) {
            self.regex_iter.nth(by - 1);
            self.consumed += by;
        }

        pub fn matches_several(&mut self, chars: &[char]) -> bool {
            for (i, expected) in chars.iter().enumerate() {
                if let Some(actual) = self.regex_iter.peek_nth(i) {
                    if actual != expected {
                        return false;
                    }
                }
            }
            return true;
        }

        pub fn consume_unseen(&mut self) -> Result<char, ParserError> {
            match self.regex_iter.next() {
                Some(c) => Ok(c),
                None => Err(ParserError::UnexexpectedEOF),
            }
        }

        pub fn matches(&mut self, expected: char) -> bool {
            if let Ok(actual) = self.peek() {
                actual == expected
            } else {
                false
            }
        }

        pub fn can_parse_group(&mut self) -> bool {
            self.matches('(')
        }

        pub fn within_bounds(&mut self) -> bool {
            self.regex_iter.peek().is_some()
        }

        pub fn can_parse_character(&mut self) -> bool {
            if let Ok(c) = self.peek() {
                !ESCAPED.contains(&c)
            } else {
                false
            }
        }

        pub fn can_parse_character_range(&mut self) -> bool {
            if let Some(c0) = self.regex_iter.peek() {
                if !ESCAPED.contains(c0) {
                    if let Some(hyphen) = self.regex_iter.peek_nth(1) {
                        if *hyphen == '-' {
                            if let Some(c1) = self.regex_iter.peek_nth(2) {
                                return !ESCAPED.contains(c1);
                            }
                        }
                    }
                }
            }
            return false;
        }

        pub fn can_parse_dot(&mut self) -> bool {
            self.matches('.')
        }

        pub fn can_parse_character_class(&mut self) -> bool {
            if let Some(c0) = self.regex_iter.peek() {
                if *c0 == '\\' {
                    if let Some(c1) = self.regex_iter.peek_nth(1) {
                        if CHARACTER_CLASSES.contains(c1) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        pub fn can_parse_escaped(&mut self) -> bool {
            if let Some(c0) = self.regex_iter.peek() {
                if *c0 == '\\' {
                    if let Some(c1) = self.regex_iter.peek_nth(1) {
                        if ESCAPED.contains(c1) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        pub fn can_parse_character_group(&mut self) -> bool {
            self.matches('[')
        }

        pub fn can_parse_match(&mut self) -> bool {
            self.can_parse_dot()
                || self.can_parse_character_class()
                || self.can_parse_character_group()
                || self.can_parse_character()
                || self.can_parse_escaped()
        }

        pub fn can_parse_anchor(&mut self) -> bool {
            if let Some(c0) = self.regex_iter.peek() {
                if *c0 == '^' || *c0 == '$' {
                    return true;
                } else if *c0 == '\\' {
                    if let Some(c1) = self.regex_iter.peek_nth(1) {
                        return ANCHORS.contains(c1);
                    }
                }
            }
            return false;
        }

        pub fn get_remainder(&mut self) -> Box<String> {
            Box::new(self.regex_iter.clone().collect::<String>())
        }

        pub fn get_consumed(&mut self) -> Box<String> {
            Box::new(self.regex.chars().take(self.consumed).collect())
        }

        pub fn can_parse_sub_expression_item(&mut self) -> bool {
            self.can_parse_group() || self.can_parse_match() || self.can_parse_anchor()
        }

        pub fn can_parse_quantifier(&mut self) -> bool {
            match self.peek() {
                Ok(c) => match c {
                    '+' | '*' | '?' | '{' => true,
                    _ => false,
                },
                Err(_) => false,
            }
        }
    }
}

fn is_word_character(char_literal: &char) -> bool {
    *char_literal == '_' || char_literal.is_ascii_alphabetic()
}
fn is_word_boundary(text: &Vec<char>, pos: usize) -> bool {
    if text.is_empty() {
        panic!("implementation error, text should never be empty")
    } else {
        let case1 = pos == 0 && is_word_character(&text[pos]);
        let case2 = pos == text.len() && is_word_character(&text[pos - 1]);
        let case3 =
            pos < text.len() && (is_word_character(&text[pos - 1]) ^ is_word_character(&text[pos]));
        return case1 || case2 || case3;
    }
}

#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub enum UpperBound {
    Undefined,
    Unbounded,
    Bounded(u64),
}

#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub enum Quantifier {
    OneOrMore(bool),
    ZeroOrMore(bool),
    ZeroOrOne(bool),
    Range(u64, UpperBound, bool),
    None,
}

impl Display for Quantifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OneOrMore(lazy) => write!(f, "+{}", if *lazy { "?" } else { "" }),
            Self::ZeroOrMore(lazy) => write!(f, "*{}", if *lazy { "?" } else { "" }),
            Self::ZeroOrOne(lazy) => write!(f, "?{}", if *lazy { "?" } else { "" }),
            Self::Range(n, m, lazy) => {
                write!(f, "{{{},{:?}}}{}", n, m, if *lazy { "?" } else { "" })
            }
            Self::None => write!(f, ""),
        }
    }
}

#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub enum Node {
    Character(char),
    Match(Box<Node>, Quantifier),
    Expression(Vec<Box<Node>>, Option<Box<Node>>),
    Group(Box<Node>, Option<usize>, Quantifier),
    Dot,
    CharacterGroup(Vec<Box<Node>>, bool),
    CharacterRange(char, char),
    // anchors
    Epsilon,
    GroupLink,
    GroupEntry(usize),
    GroupExit(usize),
    StartOfString,
    EndOfString,
    EmptyString,
    WordBoundary,
    NonWordBoundary,
    StartOfStringOnly,
    EndOfStringOnlyNotNewline,
    EndOfStringOnlyMaybeNewLine,
}

pub(crate) trait Data {
    fn accept<V: Visitor>(&self, visitor: &mut V) -> V::Result;
}

impl Node {
    pub fn accepts(&self, cursor: &Cursor, context: &Context) -> bool {
        match self {
            Node::Character(char_literal) => {
                if cursor.position < context.text.len() {
                    if context.flags.intersects(RegexFlags::IGNORECASE) {
                        char_literal.eq_ignore_ascii_case(&context.text[cursor.position])
                    } else {
                        char_literal.eq(&context.text[cursor.position])
                    }
                } else {
                    false
                }
            }
            Node::Dot => {
                cursor.position < context.text.len()
                    && (context.flags.intersects(RegexFlags::DOTALL)
                        || context.text[cursor.position] != '\n')
            }
            Node::CharacterRange(start, end) => {
                if cursor.position < context.text.len() {
                    if context.flags.intersects(RegexFlags::IGNORECASE) {
                        start
                            .to_ascii_lowercase()
                            .le(&context.text[cursor.position])
                            && context.text[cursor.position].to_ascii_lowercase().le(end)
                    } else {
                        start.le(&context.text[cursor.position])
                            && context.text[cursor.position].le(end)
                    }
                } else {
                    false
                }
            }
            Node::CharacterGroup(nodes, negated) => {
                if cursor.position < context.text.len() {
                    negated ^ nodes.iter().any(|node| node.accepts(cursor, context))
                } else {
                    false
                }
            }
            // anchors
            Node::EmptyString | Node::GroupEntry(_) | Node::GroupExit(_) => true,
            Node::WordBoundary => {
                !context.text.is_empty() && is_word_boundary(&context.text, cursor.position)
            }
            Node::NonWordBoundary => {
                !context.text.is_empty() && !is_word_boundary(&context.text, cursor.position)
            }
            Node::StartOfString => {
                let pos = cursor.position;
                pos == 0
                    || (context.flags.intersects(RegexFlags::MULTILINE)
                        && pos > 0
                        && context.text[pos - 1] == '\n')
            }
            Node::EndOfString => {
                (cursor.position >= context.text.len()
                    || (cursor.position == context.text.len() - 1
                        && context.text[cursor.position - 1] == '\n'))
                    || (context.flags.intersects(RegexFlags::MULTILINE)
                        && (cursor.position < context.text.len()
                            && context.text[cursor.position] == '\n'))
            }
            Node::StartOfStringOnly => cursor.position == 0,
            Node::EndOfStringOnlyNotNewline => cursor.position >= context.text.len(),
            Node::EndOfStringOnlyMaybeNewLine => {
                (cursor.position >= context.text.len())
                    || (cursor.position == context.text.len() - 1
                        && context.text[cursor.position] == '\n')
            }
            Node::Epsilon | Node::GroupLink => false,
            Node::Match(_, _) | Node::Expression(_, _) | Node::Group(_, _, _) => {
                panic!("accept not implemented for {:?}!", self)
            }
        }
    }

    pub fn increment(&self) -> usize {
        match self {
            Node::Character(_) | Node::Dot | Node::CharacterGroup(_, _) => 1,
            // anchors
            Node::EmptyString
            | Node::GroupEntry(_)
            | Node::GroupExit(_)
            | Node::WordBoundary
            | Node::NonWordBoundary
            | Node::StartOfString
            | Node::EndOfString
            | Node::StartOfStringOnly
            | Node::EndOfStringOnlyNotNewline
            | Node::EndOfStringOnlyMaybeNewLine
            | Node::Epsilon
            | Node::GroupLink => 0,
            _ => panic!("increment not implemented!"),
        }
    }
}

impl Data for Node {
    fn accept<V: Visitor>(&self, visitor: &mut V) -> V::Result {
        match self {
            Self::Character(_) => visitor.visit_character(self.clone()),
            Self::Expression(_, _) => visitor.visit_expression(self.clone()),
            Self::Match(_, _) => visitor.visit_match(self.clone()),
            Self::Group(_, _, _) => visitor.visit_group(self.clone()),
            Self::Dot => visitor.visit_dot(self.clone()),
            Self::CharacterGroup(_, _) => visitor.visit_character_group(self.clone()),
            Self::EmptyString
            | Self::Epsilon
            | Self::EndOfString
            | Self::EndOfStringOnlyMaybeNewLine
            | Self::EndOfStringOnlyNotNewline
            | Self::GroupEntry(_)
            | Self::GroupExit(_)
            | Self::GroupLink
            | Self::StartOfStringOnly
            | Self::StartOfString
            | Self::WordBoundary
            | Self::NonWordBoundary => visitor.visit_anchor(self.clone()),
            Self::CharacterRange(_, _) => panic!("not implemented for char range!"),
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.clone() {
            Self::Character(char_literal) => write!(f, "{}", char_literal),
            Self::Expression(items, alternative) => {
                let concated = items.iter().map(|node| format!("{}", node)).join("");
                if alternative.is_some() {
                    write!(f, "{}|{}", concated, alternative.unwrap())
                } else {
                    write!(f, "{}", concated)
                }
            }
            Self::Match(item, quantifier) => write!(f, "{}{}", item, quantifier),
            Self::Group(item, group_index, quantifier) => match group_index {
                Some(_) => write!(f, "(?:{}){}", item, quantifier),
                None => write!(f, "({}){}", item, quantifier),
            },
            Self::Epsilon => write!(f, "{}", 'ε'),
            Self::CharacterGroup(items, negated) => {
                let concated = items.iter().map(|node| format!("{}", node)).join("");
                if negated {
                    write!(f, "[^{}]", concated)
                } else {
                    write!(f, "[{}]", concated)
                }
            }
            Self::EmptyString
            | Self::Dot
            | Self::EndOfString
            | Self::EndOfStringOnlyMaybeNewLine
            | Self::EndOfStringOnlyNotNewline
            | Self::GroupEntry(_)
            | Self::GroupExit(_)
            | Self::GroupLink
            | Self::StartOfStringOnly
            | Self::StartOfString
            | Self::WordBoundary
            | Self::NonWordBoundary => write!(f, "{:?}", *self),
            Self::CharacterRange(from, to) => write!(f, "{from}-{to}",),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ParserError {
    UnexpectedToken(Box<String>, char),
    UnexexpectedEOF,
    UnableToParseChar(Box<String>),
    CantParseCharGroup(Box<String>, Box<String>),
    UnrecognizedAnchor(Box<String>, char),
    UnrecognizedModifier(Box<String>, char),
    InvalidExpression(Box<String>),
    InvalidStartToCharacterClass(Box<String>),
    SuffixRemaining(Box<String>),
    UnrecognizedQuantifier(char),
    InvalidRangeQuantifier(u64, u64),
    CantParseRangeBound(ParseIntError),
    InvalidCharacterRange(char, char),
}

impl Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::CantParseCharGroup(ref consumed, ref left) => {
                write!(
                    f,
                    "{} {}:\n | {}{}\n | {}",
                    format!("[{:0>3}]", 2).red().bold(),
                    "error while parsing character group",
                    consumed,
                    left,
                    "^".repeat(consumed.len()).green()
                )
            }
            _ => write!(f, "{:#?}", *self),
        }
    }
}

impl Error for ParserError {}

pub fn run_parse<'a>(input: &'a str, flags: &mut RegexFlags) -> Result<Node, ParserError> {
    if input.is_empty() {
        Ok(Node::EmptyString)
    } else {
        let mut parser = Parser::new(input);
        parse_inline_modifiers(&mut parser, flags)?;
        if let Ok(_) = parser.consume('^') {
            let anchor = Node::StartOfString;
            if parser.within_bounds() {
                let mut expr = parse_expression(&mut parser)?;
                if let Node::Expression(ref mut subexpressions, _) = expr {
                    subexpressions.insert(0, Box::new(anchor));
                    if parser.within_bounds() {
                        Err(ParserError::SuffixRemaining(parser.get_remainder()))
                    } else {
                        Ok(expr)
                    }
                } else {
                    panic!("expected an expression")
                }
            } else {
                Ok(anchor)
            }
        } else {
            // assert the node returned is an expression
            let expr = parse_expression(&mut parser)?;
            if parser.within_bounds() {
                Err(ParserError::SuffixRemaining(parser.get_remainder()))
            } else {
                Ok(expr)
            }
        }
    }
}

fn parse_inline_modifiers(
    parser: &mut Parser,
    flags: &mut RegexFlags,
) -> Result<bool, ParserError> {
    const ALLOWED: &[char; 4] = &['i', 'm', 's', 'x'];
    let mut modifiers: Vec<char> = Vec::new();
    while parser.matches_several(&['(', '?']) {
        parser.advance_by(2);
        while matches!(parser.peek(), Ok(c) if ALLOWED.contains(&c)) {
            parser.advance_by(1);
            modifiers.push(parser.peek().unwrap());
        }
    }
    match parser.consume(')') {
        Ok(_) => {
            modifiers.iter().for_each(|c| match c {
                'i' => *flags = *flags | RegexFlags::IGNORECASE,
                's' => *flags = *flags | RegexFlags::DOTALL,
                'm' => *flags = *flags | RegexFlags::MULTILINE,
                'x' => *flags = *flags | RegexFlags::FREESPACING,
                _ => panic!("unreachable code"),
            });
            Ok(true)
        }
        Err(err) => {
            if modifiers.is_empty() {
                Ok(true)
            } else {
                Err(err)
            }
        }
    }
}

fn parse_expression(parser: &mut Parser) -> Result<Node, ParserError> {
    let mut items: Vec<Box<Node>> = Vec::new();
    while parser.can_parse_sub_expression_item() {
        items.push(Box::new(parse_sub_expression_item(parser)?));
    }
    if items.is_empty() {
        return Err(ParserError::InvalidExpression(parser.get_remainder()));
    }
    if parser.matches('|') {
        parser.advance_by(1);
        return if parser.can_parse_sub_expression_item() {
            Ok(Node::Expression(
                items,
                Some(Box::new(parse_expression(parser)?)),
            ))
        } else {
            Ok(Node::EmptyString)
        };
    } else {
        return Ok(Node::Expression(items, None));
    }
}

fn parse_character_class(parser: &mut Parser) -> Result<Node, ParserError> {
    parser.consume('\\')?;
    return match parser.consume_unseen()? {
        'w' => Ok(Node::CharacterGroup(
            vec![
                Box::new(Node::CharacterRange('0', '9')),
                Box::new(Node::CharacterRange('A', 'z')),
                Box::new(Node::CharacterRange('a', 'z')),
                Box::new(Node::Character('-')),
            ],
            false,
        )),
        'W' => Ok(Node::CharacterGroup(
            vec![
                Box::new(Node::CharacterRange('0', '9')),
                Box::new(Node::CharacterRange('A', 'z')),
                Box::new(Node::CharacterRange('a', 'z')),
                Box::new(Node::Character('-')),
            ],
            true,
        )),
        'd' => Ok(Node::CharacterGroup(
            vec![Box::new(Node::CharacterRange('0', '9'))],
            false,
        )),
        'D' => Ok(Node::CharacterGroup(
            vec![Box::new(Node::CharacterRange('0', '9'))],
            true,
        )),
        's' => Ok(Node::CharacterGroup(
            vec![' ', '\t', '\n']
                .iter()
                .map(|literal| Box::new(Node::Character(*literal)))
                .collect_vec(),
            false,
        )),
        'S' => Ok(Node::CharacterGroup(
            vec![' ', '\t', '\n']
                .iter()
                .map(|literal| Box::new(Node::Character(*literal)))
                .collect_vec(),
            true,
        )),
        char_literal => Err(ParserError::UnrecognizedAnchor(
            parser.get_remainder(),
            char_literal,
        )),
    };
}

fn parse_character_range(parser: &mut Parser) -> Result<Node, ParserError> {
    let start = parser.consume_unseen()?;
    parser.consume('-')?;
    let end = parser.consume_unseen()?;

    if start > end {
        Err(ParserError::InvalidCharacterRange(start, end))
    } else {
        Ok(Node::CharacterRange(start, end))
    }
}

fn parse_character_group_item(parser: &mut Parser) -> Result<Node, ParserError> {
    if parser.can_parse_character_class() {
        parse_character_class(parser)
    } else {
        if parser.can_parse_character_range() {
            parse_character_range(parser)
        } else {
            parse_character_in_character_group(parser)
        }
    }
}

fn parse_character_group(parser: &mut Parser) -> Result<Node, ParserError> {
    parser.consume('[')?;
    let mut negated = false;
    if parser.matches('^') {
        negated = true;
        parser.advance_by(1);
    }
    let mut items: Vec<Box<Node>> = Vec::new();
    loop {
        match parse_character_group_item(parser) {
            Ok(node) => {
                items.push(Box::new(node));
            }
            Err(err) => {
                if let ParserError::UnableToParseChar(_) = err {
                    break;
                } else {
                    return Err(err);
                }
            }
        }
    }
    parser.consume(']')?;
    if items.is_empty() {
        return Err(ParserError::CantParseCharGroup(
            parser.get_consumed(),
            parser.get_remainder(),
        ));
    } else {
        return Ok(Node::CharacterGroup(items, negated));
    }
}

fn parse_escaped<'a>(parser: &mut Parser) -> Result<Node, ParserError> {
    parser.consume('\\')?;
    Ok(Node::Character(parser.consume_unseen()?))
}

fn parse_character(parser: &mut Parser) -> Result<Node, ParserError> {
    if parser.can_parse_escaped() {
        parse_escaped(parser)
    } else {
        if !parser.can_parse_character() {
            Err(ParserError::UnableToParseChar(parser.get_remainder()))
        } else {
            Ok(Node::Character(parser.consume_unseen()?))
        }
    }
}

fn parse_character_in_character_group(parser: &mut Parser) -> Result<Node, ParserError> {
    if parser.can_parse_escaped() {
        parse_escaped(parser)
    } else {
        if parser.matches(']') {
            Err(ParserError::UnableToParseChar(parser.get_remainder()))
        } else {
            Ok(Node::Character(parser.consume_unseen()?))
        }
    }
}

fn parse_match_item<'a>(parser: &mut Parser) -> Result<Node, ParserError> {
    if parser.matches('.') {
        parser.consume('.')?;
        Ok(Node::Dot)
    } else if parser.can_parse_character_class() {
        parse_character_class(parser)
    } else if parser.can_parse_character_group() {
        parse_character_group(parser)
    } else if parser.can_parse_group() {
        parse_character_group(parser)
    } else {
        parse_character(parser)
    }
}

fn validate_range_quantifier(
    lower: u64,
    upper: UpperBound,
    lazy: bool,
) -> Result<Quantifier, ParserError> {
    match upper {
        UpperBound::Bounded(upper_digit) => {
            if upper_digit < lower {
                Err(ParserError::InvalidRangeQuantifier(lower, upper_digit))
            } else {
                Ok(Quantifier::Range(lower, upper, lazy))
            }
        }
        _ => Ok(Quantifier::Range(lower, upper, lazy)),
    }
}

fn parse_int<'b>(parser: &mut Parser) -> Result<u64, ParserError> {
    let mut digits: Vec<char> = Vec::new();
    loop {
        if let Ok(digit) = parser.peek() {
            if digit.is_ascii_digit() {
                digits.push(digit);
                parser.advance_by(1);
                continue;
            }
        }
        break;
    }
    let number_stream: String = digits.iter().collect();
    match format!("{}", number_stream).parse::<u64>() {
        Ok(num) => Ok(num),
        Err(parse_int_error) => return Err(ParserError::CantParseRangeBound(parse_int_error)),
    }
}

fn parse_range_quantifier(parser: &mut Parser) -> Result<Quantifier, ParserError> {
    parser.consume('{')?;
    let mut lower: u64 = 0;
    if !parser.matches(',') {
        lower = parse_int(parser)?;
    }
    let mut upper = UpperBound::Undefined;
    if let Ok(_) = parser.consume(',') {
        upper = UpperBound::Unbounded;
        if let Ok(c) = parser.peek() {
            if c.is_ascii_digit() {
                upper = UpperBound::Bounded(parse_int(parser)?);
            }
        }
    }
    parser.consume('}')?;
    let mut lazy = false;
    if parser.matches('?') {
        parser.advance_by(1);
        lazy = true;
    }

    return validate_range_quantifier(lower, upper, lazy);
}

fn parse_quantifier(parser: &mut Parser) -> Result<Quantifier, ParserError> {
    if parser.matches('{') {
        parse_range_quantifier(parser)
    } else {
        let char_literal = parser.consume_unseen()?;
        match char_literal {
            '*' | '+' | '?' => {
                let lazy = if let Ok(_) = parser.consume('?') {
                    true
                } else {
                    false
                };
                match char_literal {
                    '*' => Ok(Quantifier::ZeroOrMore(lazy)),
                    '+' => Ok(Quantifier::OneOrMore(lazy)),
                    '?' => Ok(Quantifier::ZeroOrOne(lazy)),
                    _ => panic!("unrecognized quantifier {:?}", char_literal),
                }
            }
            _ => Err(ParserError::UnrecognizedQuantifier(char_literal)),
        }
    }
}

fn parse_group<'a>(parser: &mut Parser) -> Result<Node, ParserError> {
    parser.consume('(')?;

    let group_index = if parser.matches_several(&['?', ':']) {
        parser.advance_by(2);
        None
    } else {
        parser.increment_group_count();
        Some(parser.group_count() - 1)
    };
    let expression = if parser.matches('?') {
        Node::EmptyString
    } else {
        parse_expression(parser)?
    };
    parser.consume(')')?;

    let quantifier = if parser.can_parse_quantifier() {
        parse_quantifier(parser)?
    } else {
        Quantifier::None
    };
    Ok(Node::Group(Box::new(expression), group_index, quantifier))
}

fn parse_anchor<'a>(parser: &mut Parser) -> Result<Node, ParserError> {
    match parser.consume('\\') {
        Ok(_) => {
            let char_literal = parser.consume_unseen()?;
            return match char_literal {
                'a' => Ok(Node::StartOfStringOnly),
                'b' => Ok(Node::WordBoundary),
                'B' => Ok(Node::NonWordBoundary),
                'z' => Ok(Node::EndOfStringOnlyNotNewline),
                'Z' => Ok(Node::EndOfStringOnlyMaybeNewLine),
                _ => Err(ParserError::UnrecognizedAnchor(
                    parser.get_remainder(),
                    char_literal,
                )),
            };
        }
        Err(err) => match parser.peek() {
            Ok(c) if c == '^' || c == '$' => {
                parser.advance_by(1);
                if c == '^' {
                    Ok(Node::StartOfString)
                } else {
                    Ok(Node::EndOfString)
                }
            }
            _ => Err(err),
        },
    }
}

fn parse_match(parser: &mut Parser) -> Result<Node, ParserError> {
    let match_item = parse_match_item(parser)?;
    let quantifier = if parser.can_parse_quantifier() {
        parse_quantifier(parser)?
    } else {
        Quantifier::None
    };
    return Ok(Node::Match(Box::new(match_item), quantifier));
}

fn parse_sub_expression_item<'a>(parser: &mut Parser) -> Result<Node, ParserError> {
    if parser.can_parse_group() {
        parse_group(parser)
    } else if parser.can_parse_anchor() {
        parse_anchor(parser)
    } else {
        parse_match(parser)
    }
}

pub mod visitor {
    use super::Node;

    pub trait Visitor {
        type Result;
        fn visit_expression(&mut self, expression: Node) -> Self::Result;
        fn visit_character(&mut self, char: Node) -> Self::Result;
        fn visit_anchor(&mut self, anchor: Node) -> Self::Result;
        fn visit_dot(&mut self, dot: Node) -> Self::Result;
        fn visit_match(&mut self, match_: Node) -> Self::Result;
        fn visit_character_group(&mut self, character_group: Node) -> Self::Result;
        fn visit_group(&mut self, group: Node) -> Self::Result;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_new() {
        let mut p = Parser::new("abc");
        assert_eq!(p.group_count(), 0);
        assert_eq!(p.consume_unseen(), Ok('a'));
        assert_eq!(p.consume_unseen(), Ok('b'));
        assert_eq!(p.consume_unseen(), Ok('c'));
        assert_eq!(p.consume_unseen(), Err(ParserError::UnexexpectedEOF));
    }

    #[test]
    fn test_start_of_string_accept() {
        let chars: Vec<char> = String::from("abc\n").chars().collect();
        let mut cursor = Cursor::new(0, 0);
        let context = Context::new(chars.clone());
        let start_of_string = Node::StartOfString;
        assert_eq!(start_of_string.accepts(&cursor, &context), true);
        cursor = Cursor {
            position: cursor.position + 1,
            groups: cursor.groups,
        };
        assert_eq!(start_of_string.accepts(&cursor, &context), false);
        cursor = Cursor {
            position: cursor.position + 2,
            groups: cursor.groups,
        };
        assert_eq!(start_of_string.accepts(&cursor, &context), false);

        let context_with_multiline = Context::new_with_flags(chars, RegexFlags::MULTILINE);
        assert_eq!(
            start_of_string.accepts(&cursor, &context_with_multiline),
            false
        );
        cursor = Cursor {
            position: cursor.position + 1,
            groups: cursor.groups,
        };
        assert_eq!(
            start_of_string.accepts(&cursor, &context_with_multiline),
            true
        );
    }

    #[test]
    fn test_character_accept() {
        let chars: Vec<char> = String::from("abA").chars().collect();
        let mut cursor = Cursor::new(0, 0);
        let context = Context::new(chars.clone());
        let a = Node::Character('a');
        assert_eq!(a.accepts(&cursor, &context), true);
        cursor = Cursor {
            position: cursor.position + 1,
            groups: cursor.groups,
        };
        assert_eq!(a.accepts(&cursor, &context), false);
        cursor = Cursor {
            position: cursor.position + 1,
            groups: cursor.groups,
        };

        let context_with_ignorecase = Context::new_with_flags(chars, RegexFlags::IGNORECASE);
        assert_eq!(a.accepts(&cursor, &context_with_ignorecase), true);
        cursor = Cursor {
            position: cursor.position + 1,
            groups: cursor.groups,
        };
        assert_eq!(a.accepts(&cursor, &context_with_ignorecase), false);
    }

    #[test]
    fn test_dot_character_accept() {
        let chars: Vec<char> = String::from("a\n").chars().collect();
        let mut cursor = Cursor::new(0, 0);
        let context = Context::new(chars.clone());
        let dot = Node::Dot;
        assert_eq!(dot.accepts(&cursor, &context), true);
        cursor = Cursor {
            position: cursor.position + 1,
            groups: cursor.groups,
        };
        assert_eq!(dot.accepts(&cursor, &context), false);

        let context_with_ignorecase = Context::new_with_flags(chars, RegexFlags::DOTALL);
        assert_eq!(dot.accepts(&cursor, &context_with_ignorecase), true);
    }
}
