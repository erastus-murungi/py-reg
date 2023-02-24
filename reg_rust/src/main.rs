use crate::fsm::RegexNFA;

pub mod fsm;
pub mod matching;
pub mod parser;
pub mod utils;

fn main() {
    let pattern = String::from("([^/]*/)*sub1/");
    let mut regex = RegexNFA::new(&pattern);
    println!("{:#?}", regex.compile());
    println!("{:#?}", regex);
    let _ = regex.render();
}
