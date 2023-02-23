use crate::utils::RegexFlags;

pub struct Cursor {
    pub position: usize,
    pub groups: Vec<usize>,
}

impl Cursor {
    pub fn new(position: usize, n_groups: usize) -> Cursor {
        return Cursor {
            position: position,
            groups: vec![usize::MAX; n_groups * 2],
        };
    }
}

pub struct Context<'a> {
    pub text: &'a Vec<char>,
    flags: RegexFlags,
}

impl<'a> Context<'a> {
    pub fn new(text: &'a Vec<char>) -> Context {
        return Context {
            text: text,
            flags: RegexFlags::NO_FLAG,
        };
    }

    pub fn new_with_flags(text: &'a Vec<char>, flags: RegexFlags) -> Context {
        return Context {
            text: text,
            flags: flags,
        };
    }
}
