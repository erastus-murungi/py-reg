use nom::{bytes::complete::tag, IResult};

fn parse_non_capturing_group_identifier(input: &str) -> IResult<&str, &str> {
    tag(":?")(input)
}

// fn parse_group<'a>(input: &str) -> IResult<&str, &str> {
//     let (input, _) = tag("(")(input)?;
//     return match tag("?:")(input) {
//         Ok(x) => {
//             return Ok(x);
//         }
//         Err((x)) => Ok(("", "")),
//     };
//     return Ok((("", "")));
// }

#[cfg(test)]
mod tests {
    use crate::nom_parser::*;

    #[test]
    fn test_parse_non_capturing_group_identifier() {
        println!("{:?}", parse_non_capturing_group_identifier("?:"));
    }
}
