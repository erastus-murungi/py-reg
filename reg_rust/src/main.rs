use crate::fsm::RegexNFA;
use crate::parser::*;
use crate::utils::RegexFlags;

pub mod fsm;
pub mod matching;
pub mod parser;
pub mod utils;

fn main() {
    let pattern = String::from("a[0-9]\\w");
    let mut regex = RegexNFA::new(&pattern);
    println!("{:#?}", regex.compile());
    println!("{:#?}", regex);
    regex.as_graphviz_code();
}
