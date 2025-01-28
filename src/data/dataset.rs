#![allow(dead_code, unused_variables, unused_imports)]

use burn::data::dataset::{Dataset, SqliteDataset, source::huggingface::HuggingfaceDatasetLoader};

#[derive(Clone, Debug, derive_new::new, burn::serde::Serialize, burn::serde::Deserialize)]
pub struct GoItem {
    pub __key__: String,
    pub __url__: String,
    pub board: Vec<u8>,
}

impl GoItem {
    fn output_count() -> usize {
        return 9 * 9;
    }
}

pub struct GoDataset {
    dataset: SqliteDataset<GoItem>,
}

impl GoDataset {
    pub fn train() -> Self {
        let dataset: SqliteDataset<GoItem> = HuggingfaceDatasetLoader::new("go-dataset-9x9")
            .dataset("train")
            .unwrap();
        Self { dataset }
    }

    pub fn test() -> Self {
        let dataset: SqliteDataset<GoItem> = HuggingfaceDatasetLoader::new("go-dataset-9x9")
            .dataset("test")
            .unwrap();
        Self { dataset }
    }
}

impl Dataset<GoItem> for GoDataset {
    fn get(&self, index: usize) -> Option<GoItem> {
        self.dataset
            .get(index)
            .map(|item| GoItem::new(item.__key__, item.__url__, item.board))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
