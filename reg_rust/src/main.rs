use crate::matching::{Context, Cursor};
use crate::parse::*;
use crate::utils::RegexFlags;

pub mod fsm;
pub mod matching;
pub mod parse;
pub mod utils;

fn main() {
    let pattern = String::from("\\]\\[");
    let mut flags = RegexFlags::NO_FLAG;

    let k: Vec<char> = pattern.chars().collect();
    let result = run_parse(&k, &mut flags);
    println!("{:#?}", result);

    println!("{:?}", flags);
}
