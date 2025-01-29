use crate::{
    data::{batcher::GoBatcher, dataset::GoDataset},
    model::Model,
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::{AdamConfig, decay::WeightDecayConfig},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};
use serde::{Deserialize, Serialize};
use serde_json;
use std::{fs, io::Write};

#[derive(Deserialize, Serialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub workers: usize,
    pub seed: u64,
    pub optimizer: AdamConfig,
}

impl TrainingConfig {
    pub fn new(
        epochs: usize,
        batch_size: usize,
        workers: usize,
        seed: u64,
        optimizer: AdamConfig,
    ) -> Self {
        return Self {
            epochs,
            batch_size,
            workers,
            seed,
            optimizer,
        };
    }
}

pub fn run_train<B: AutodiffBackend>(device: B::Device) {
    const MODEL_DIR: &str = "./mnist-model";
    fs::remove_dir_all(MODEL_DIR).ok();
    fs::create_dir_all(MODEL_DIR).ok();

    let train_config = TrainingConfig::new(
        24,
        32,
        5,
        42,
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-5))),
    );
    B::seed(train_config.seed);
    let load_train_data = DataLoaderBuilder::new(GoBatcher::<B>::new(device.clone()))
        .batch_size(train_config.batch_size)
        .shuffle(train_config.seed)
        .num_workers(train_config.workers)
        .build(GoDataset::train());

    let load_test_data = DataLoaderBuilder::new(GoBatcher::<B::InnerBackend>::new(device.clone()))
        .batch_size(train_config.batch_size)
        .shuffle(train_config.seed)
        .num_workers(train_config.workers)
        .build(GoDataset::test());

    let trained_model = LearnerBuilder::new(MODEL_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(train_config.epochs)
        .summary()
        .build(Model::new(&device), train_config.optimizer.init(), 1e-4)
        .fit(load_train_data, load_test_data);

    let jsonify_config = serde_json::to_string(&train_config);

    match jsonify_config {
        Ok(jsonified_config) => {
            let mut file = std::fs::File::create(format!("{}/config.json", MODEL_DIR))
                .expect("Failed to create or open file");

            let _ = file.flush();
            let _ = file
                .write_all(format!("{}", jsonified_config).as_bytes())
                .expect("Failed write");
        }
        Err(err) => eprintln!("{}", err),
    }

    trained_model
        .save_file(
            format!("{}/model", MODEL_DIR),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save model");
}
