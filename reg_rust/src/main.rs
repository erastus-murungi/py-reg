use crate::matching::{Context, Cursor};
use crate::parser::*;
use crate::utils::RegexFlags;

pub mod fsm;
pub mod matching;
pub mod parser;
pub mod utils;

fn main() {
    let pattern = String::from("[]]ab{10,2}c");
    let mut flags = RegexFlags::NO_FLAG;

    let result = run_parse(pattern.as_str(), &mut flags);
    println!("{:#?}", result);

    println!("{:?}", flags);
}
