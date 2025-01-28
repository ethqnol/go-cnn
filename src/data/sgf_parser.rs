#![allow(dead_code, unused_imports, unused_variables)]

use std::error::Error;

static BOARD_SIZE: usize = 9;

pub struct ParseError {
    note: String,
}

pub fn sgf_parser(sgf_ascii: &Vec<u8>) -> Result<(Vec<u8>, usize), ParseError> {
    let sgf = std::str::from_utf8(&sgf_ascii[5..BOARD_SIZE]).expect("invalid utf-8 sequence");
    let mut board = vec![0u8; BOARD_SIZE * BOARD_SIZE];

    let mut current_player = 1;

    let moves = parse_moves(sgf);

    for i in 0..moves.len() - 1 {
        let (x, y) = moves[i];
        if x < BOARD_SIZE && y < BOARD_SIZE {
            board[encode_to_1d((x, y))] = current_player;
            current_player = if current_player == 1 { 2 } else { 1 };
        } else {
            return Err(ParseError {
                note: "Poorly formatted board".to_string(),
            });
        }
    }
    let last_move = moves[moves.len() - 1];
    if last_move.0 < BOARD_SIZE || last_move.1 < BOARD_SIZE {
        return Err(ParseError {
            note: "Poorly formatted board".to_string(),
        });
    }
    return Ok((board, encode_to_1d(last_move)));
}

fn parse_moves(sgf: &str) -> Vec<(usize, usize)> {
    let mut moves = Vec::new();

    for token in sgf.split(';') {
        if token.starts_with("B[") || token.starts_with("W[") {
            if let Some(coords) = token.get(2..4) {
                if coords.len() == 2 {
                    let x = (coords.chars().nth(0).unwrap() as usize) - ('a' as usize);
                    let y = (coords.chars().nth(1).unwrap() as usize) - ('a' as usize);
                    moves.push((x, y));
                }
            }
        }
    }

    moves
}

pub fn encode_to_1d(loc: (usize, usize)) -> usize {
    return loc.0 * BOARD_SIZE + loc.1;
}

pub fn decode_to_2d(loc: usize) -> (usize, usize) {
    return (loc % BOARD_SIZE, loc / BOARD_SIZE);
}
