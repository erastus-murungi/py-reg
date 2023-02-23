use crate::matching::{Context, Cursor};
use crate::parse::*;

pub mod fsm;
pub mod matching;
pub mod parse;

fn main() {
    let pattern = String::from("(?:[a-z]+? | b)");

    let k: Vec<char> = pattern.chars().collect();
    let result = run_parse(&k);
    println!("{:#?}", result);
    let a = Box::new(Node::Character('T'));
    let b = Quantifier::OneOrMore(true);
    let c = Node::Match(a.clone(), b);

    println!("{:?}", c);
    // let nfa = RegexNFA::new(&k);
    // println!("nfa {:?}", nfa);
    // let char = Character::new('a');
    // println!("{:?}", char);
    let cursor = Cursor::new(0, 0);
    let context = Context::new(&k);
    println!("accepts {:?}", a.accepts(cursor, context));
    // println!()
}
