#![allow(unused_variables, dead_code, unused_imports)]

use crate::data::dataset::*;
use crate::data::sgf_parser::*;
use burn::{data::dataloader::batcher::Batcher, prelude::*};
#[derive(Clone, derive_new::new)]
pub struct GoBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct GoBatch<B: Backend> {
    pub board_states: Tensor<B, 4, Int>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<GoItem, GoBatch<B>> for GoBatcher<B> {
    fn batch(&self, items: Vec<GoItem>) -> GoBatch<B> {
        let (board_states, targets): (Vec<Vec<u8>>, Vec<usize>) = items
            .iter()
            .filter_map(|item| {
                let parser_result = sgf_parser(&item.board);
                match parser_result {
                    Ok((board_state, expected_move)) => return Some((board_state, expected_move)),
                    Err(_) => None,
                }
            })
            .collect();

        let board_states: Tensor<B, 4, Int> = Tensor::stack(
            board_states
                .iter()
                .map(|item| TensorData::new(item.clone(), Shape::new([9, 9, 1])))
                .map(|tensor_data| {
                    Tensor::<B, 3, Int>::from_data(tensor_data, &self.device).permute([2, 0, 1])
                })
                .collect(),
            0,
        );

        let targets = Tensor::cat(
            targets
                .iter()
                .map(|item| {
                    Tensor::<B, 1, Int>::from_data(
                        TensorData::from([(item.clone() as i64).elem::<B::IntElem>()]),
                        &self.device,
                    )
                })
                .collect(),
            0,
        );

        return GoBatch {
            board_states,
            targets,
        };
    }
}
