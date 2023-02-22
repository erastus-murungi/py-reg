pub type State = u64;

#[derive(PartialEq, Debug)]
pub struct Environment {
    state_counter: u64,
}

impl Environment {
    pub fn new() -> Environment {
        Environment { state_counter: 0 }
    }

    pub fn gen_state(&mut self) -> State {
        self.state_counter += 1;
        return self.state_counter;
    }
}

#[derive(PartialEq, Debug)]
pub struct RegexNFA<'a> {
    pattern: &'a Vec<char>,
}

impl<'a> RegexNFA<'a> {
    pub fn new(pattern: &Vec<char>) -> RegexNFA {
        RegexNFA { pattern }
    }
}
