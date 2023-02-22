use fsm::Environment;
use fsm::RegexNFA;
use parse::Parser;

use crate::matching::{Context, Cursor};
use crate::parse::Character;
use crate::parse::Matcher;

pub mod fsm;
pub mod matching;
pub mod parse;

fn main() {
    let environ = Environment::new();
    let pattern = String::from("This is the string that I will be parsing ");

    let k: Vec<char> = pattern.chars().collect();
    let parser = Parser::new(&k);
    // let nfa = RegexNFA::new(&k);
    // println!("nfa {:?}", nfa);
    // let char = Character::new('a');
    // println!("{:?}", char);
    // let cursor = Cursor::new(0, 0);
    // let context = Context::new(&k);
    // println!("accepts {:?}", char.accepts(cursor, context));
    // println!()

    println!("{:?}", parser);
}
