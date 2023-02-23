use itertools::Itertools;

use crate::{
    matching::{Context, Cursor},
    utils::RegexFlags,
};
use core::panic;
use std::{hash::Hash, num::ParseIntError};

#[derive(Debug)]
pub struct Parser<'a> {
    group_count: u64,
    position: usize,
    regex: &'a Vec<char>,
}

#[derive(Debug, Hash, Clone)]
pub enum UpperBound {
    Undefined,
    Unbounded,
    Bounded(u64),
}

#[derive(Debug, Hash, Clone)]
pub enum Quantifier {
    OneOrMore(bool),
    ZeroOrMore(bool),
    ZeroOrOne(bool),
    Range(u64, UpperBound, bool),
    None,
}

#[derive(Debug, Hash, Clone)]
pub enum Node {
    Character(char),
    Match(Box<Node>, Quantifier),
    Expression(Vec<Box<Node>>, Option<Box<Node>>),
    Group(Box<Node>, Option<u64>, Quantifier),
    AnyCharacter,
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

impl Node {
    pub fn accepts(&self, cursor: Cursor, context: Context) -> bool {
        match self {
            Node::Character(char_literal) => *char_literal == context.text[cursor.position],
            Node::Match(_, _) => todo!(),
            Node::Expression(_, _) => todo!(),
            Node::Group(_, _, _) => todo!(),
            Node::AnyCharacter => todo!(),
            Node::CharacterGroup(_, _) => todo!(),
            // anchors
            Node::EmptyString => cursor.position == 0,
            Node::GroupEntry(_) => true,
            Node::GroupExit(_) => true,
            Node::WordBoundary => true,
            Node::NonWordBoundary => true,
            Node::StartOfString => true,
            Node::EndOfString => true,
            Node::StartOfStringOnly => cursor.position == 0,
            Node::EndOfStringOnlyNotNewline => cursor.position >= context.text.len(),
            Node::EndOfStringOnlyMaybeNewLine => true,
            Node::Epsilon => false,
            Node::GroupLink => false,
            Node::CharacterRange(_, _) => panic!("char range not implemented!"),
        }
    }

    pub fn increment(&self) -> usize {
        match self {
            Node::Character(_) | Node::AnyCharacter | Node::CharacterGroup(_, _) => 1,
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

// we take a parsing state and return either a valid node or an error

static ESCAPED: &'static [char] = &[
    '$', '(', ')', '*', '+', '-', '.', '<', '=', '>', '?', '[', '\\', ']', '^', '{', '|', '}',
];

static CHARACTER_CLASSES: &'static [char] = &['w', 'W', 's', 'S', 'd', 'D'];
static ANCHORS: &'static [char] = &['A', 'z', 'Z', 'G', 'b', 'B'];

impl<'a> Parser<'a> {
    pub fn new(regex: &Vec<char>) -> Parser {
        Parser {
            group_count: 0,
            position: 0,
            regex: regex,
        }
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn advance_by(&mut self, by: usize) {
        self.position += by;
    }

    fn peek(&self) -> Result<char, ParserError> {
        if self.within_bounds() {
            Ok(self.regex[self.position])
        } else {
            Err(ParserError::UnexexpectedEOF)
        }
    }

    fn consume(&mut self, c: char) -> Result<char, ParserError> {
        let position = self.position;
        let regex = self.regex;

        if position >= regex.len() {
            return Err(ParserError::UnexexpectedEOF);
        } else if regex[position] != c {
            return Err(ParserError::UnexpectedToken(position, c, regex[position]));
        } else {
            self.advance();
            return Ok(c);
        }
    }

    fn matches_several(&self, chars: &[char]) -> bool {
        if self.position + chars.len() < self.regex.len() {
            for (index, char) in chars.iter().enumerate() {
                if *char != self.regex[self.position + index] {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    fn consume_unseen(&mut self) -> Result<char, ParserError> {
        let position = self.position;
        let regex = self.regex;

        if position >= regex.len() {
            return Err(ParserError::UnexexpectedEOF);
        } else {
            let c = self.regex[self.position];
            self.advance();
            return Ok(c);
        }
    }

    pub fn matches(&self, c: char) -> bool {
        self.within_bounds() && self.regex[self.position] == c
    }

    pub fn matches_ahead(&self, c: char, offset: usize) -> bool {
        self.within_bounds_by(offset) && self.regex[self.position + offset] == c
    }

    pub fn can_parse_group(&self) -> bool {
        self.matches('(')
    }

    pub fn within_bounds(&self) -> bool {
        self.position < self.regex.len()
    }

    pub fn within_bounds_by(&self, by: usize) -> bool {
        self.position + by < self.regex.len()
    }

    pub fn can_parse_character(&self) -> bool {
        self.within_bounds() && !ESCAPED.contains(&self.regex[self.position])
    }

    pub fn can_parse_dot(&self) -> bool {
        self.matches('.')
    }

    pub fn can_parse_character_class(&self) -> bool {
        self.matches('\\')
            && self.position + 1 < self.regex.len()
            && CHARACTER_CLASSES.contains(&self.regex[self.position + 1])
    }

    pub fn can_parse_escaped(&self) -> bool {
        self.matches('\\')
            && self.position + 1 < self.regex.len()
            && ESCAPED.contains(&self.regex[self.position + 1])
    }

    pub fn can_parse_character_group(&self) -> bool {
        self.matches('[')
    }

    pub fn can_parse_match(&self) -> bool {
        self.can_parse_dot()
            || self.can_parse_character_class()
            || self.can_parse_character_group()
            || self.can_parse_character()
            || self.can_parse_escaped()
    }

    pub fn can_parse_anchor(&self) -> bool {
        self.matches('^')
            || self.matches('$')
            || (self.matches('\\')
                && self.within_bounds_by(1)
                && ANCHORS.contains(&self.regex[self.position + 1]))
    }

    pub fn can_parse_sub_expression_item(&self) -> bool {
        self.can_parse_group() || self.can_parse_match() || self.can_parse_anchor()
    }

    pub fn can_parse_quantifier(&self) -> bool {
        match self.peek() {
            Ok(c) => match c {
                '+' | '*' | '?' | '{' => true,
                _ => false,
            },
            Err(_) => false,
        }
    }
}

#[derive(Debug)]
pub enum ParserError {
    UnexpectedToken(usize, char, char),
    UnexexpectedEOF,
    UnableToParseChar(usize),
    CantParseCharGroup(usize),
    UnrecognizedAnchor(usize, char),
    UnrecognizedModifier(usize, char),
    InvalidExpression(usize),
    InvalidStartToCharacterClass(usize),
    SuffixRemaining(usize),
    UnrecognizedQuantifier(char),
    InvalidRangeQuantifier(u64, u64),
    CantParseRangeBound(ParseIntError),
    InvalidCharacterRange(char, char),
}

pub fn run_parse<'a>(regex: &'a Vec<char>, flags: &mut RegexFlags) -> Result<Node, ParserError> {
    if regex.is_empty() {
        return Ok(Node::EmptyString);
    } else {
        let mut parser = Parser::new(regex);
        parse_inline_modifiers(&mut parser, flags)?;
        if let Ok(_) = parser.consume('^') {
            let anchor = Node::StartOfString;
            if parser.within_bounds() {
                let mut expr = parse_expression(&mut parser)?;
                if let Node::Expression(ref mut subexpressions, _) = expr {
                    subexpressions.insert(0, Box::new(anchor));
                    if parser.within_bounds() {
                        return Err(ParserError::SuffixRemaining(parser.position));
                    }
                    return Ok(expr);
                } else {
                    panic!("expected an expression")
                }
            } else {
                return Ok(anchor);
            }
        } else {
            // assert the node returned is an expression
            let expr = parse_expression(&mut parser)?;
            if parser.within_bounds() {
                return Err(ParserError::SuffixRemaining(parser.position));
            } else {
                return Ok(expr);
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
        loop {
            match parser.peek() {
                Ok(c) if ALLOWED.contains(&c) => {
                    parser.advance();
                    modifiers.push(c)
                }
                _ => break,
            }
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
        return Err(ParserError::InvalidExpression(parser.position));
    }
    if parser.matches('|') {
        parser.advance();
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
            parser.position,
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
        if parser.within_bounds()
            && !ESCAPED.contains(&parser.regex[parser.position])
            && parser.matches_ahead('-', 1)
            && parser.within_bounds_by(2)
            && !ESCAPED.contains(&parser.regex[parser.position + 2])
        {
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
        parser.advance();
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
        return Err(ParserError::CantParseCharGroup(parser.position));
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
            Err(ParserError::UnableToParseChar(parser.position))
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
            Err(ParserError::UnableToParseChar(parser.position))
        } else {
            Ok(Node::Character(parser.consume_unseen()?))
        }
    }
}

fn parse_match_item<'a>(parser: &mut Parser) -> Result<Node, ParserError> {
    if parser.matches('.') {
        parser.consume('.')?;
        Ok(Node::AnyCharacter)
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

fn parse_range_quantifier(parser: &mut Parser) -> Result<Quantifier, ParserError> {
    parser.consume('{')?;
    let mut lower: u64 = 0;
    if !parser.matches(',') {
        let number_stream: String = parser.regex[parser.position..]
            .iter()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        parser.position += number_stream.len();
        match format!("{}", number_stream).parse() {
            Ok(num) => lower = num,
            Err(parse_int_error) => return Err(ParserError::CantParseRangeBound(parse_int_error)),
        }
    }
    let mut upper = UpperBound::Undefined;
    if parser.matches(',') {
        upper = UpperBound::Unbounded;
        parser.advance();
        if parser.regex[parser.position].is_ascii_digit() {
            let number_stream: String = parser.regex[parser.position..]
                .iter()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            parser.position += number_stream.len();
            match format!("{}", number_stream).parse() {
                Ok(num) => upper = UpperBound::Bounded(num),
                Err(parse_int_error) => {
                    return Err(ParserError::CantParseRangeBound(parse_int_error))
                }
            }
        }
    }
    parser.consume('}')?;
    let mut lazy = false;
    if parser.matches('?') {
        parser.advance();
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
        parser.group_count += 1;
        Some(parser.group_count)
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
                    parser.position,
                    char_literal,
                )),
            };
        }
        Err(err) => match parser.peek() {
            Ok(c) if c == '^' || c == '$' => {
                parser.advance();
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

fn parse_match<'a>(parser: &mut Parser) -> Result<Node, ParserError> {
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
