use std::fs;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use reg_rust::fsm::RegexNFA;
use reg_rust::matching::Matcher;

fn do_the_work(text: &str, expected: Vec<&str>) {
    let pattern = String::from(r"[\w\.-]+@([\w-]+\.)+[\w-]{2,4}");
    let mut reg = RegexNFA::new(pattern.as_str());
    reg.compile().unwrap();
    let actual: Vec<String> = reg.find_iter(text).map(|m| m.as_str()).collect();
    assert_eq!(expected, actual)
}

fn criterion_benchmark_regex_nfa(c: &mut Criterion) {
    let file_path = "../benchmark/input-text.txt";
    println!("In file {}", file_path);

    let contents: String = fs::read_to_string(file_path)
        .expect("Expected have been able to read the file")
        .chars()
        .take(200000)
        .collect();
    let expected: Vec<&str> = regex::Regex::new(r"[\w\.-]+@([\w-]+\.)+[\w-]{2,4}")
        .unwrap()
        .find_iter(&contents)
        .map(|m| m.as_str())
        .collect();
    c.bench_function("parse emails large file", |b| {
        b.iter(|| do_the_work(black_box(&contents), black_box(expected.clone())))
    });
}

criterion_group!(benches, criterion_benchmark_regex_nfa);
criterion_main!(benches);
