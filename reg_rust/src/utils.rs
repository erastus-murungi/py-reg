use bitflags::bitflags;

bitflags! {
    pub struct RegexFlags: u32 {
        const NO_FLAG = 0;
        const NO_BACKTRACK = 1 << 1;
        const IGNORECASE = 1 << 2;
        const MULTILINE = 1 << 3;
        const DOTALL = 1 << 4;
        const FREESPACING = 1 << 5;
        const OPTIMIZE = 1 << 6;
        const DEBUG = 1 << 7;
    }
}
