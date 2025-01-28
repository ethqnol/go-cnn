use crate::data::batcher::GoBatch;
use burn::{
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLossConfig,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    //conv layers
    conv_1: ConvBlock<B>,
    conv_2: ConvBlock<B>,
    conv_3: ConvBlock<B>,
    conv_4: ConvBlock<B>,
    conv_5: ConvBlock<B>,
    //fc layers
    dropout_1: Dropout,
    fc_1: LinearBlock<B>,
    dropout_2: Dropout,
    fc_2: LinearBlock<B>,
    dropout_3: Dropout,
    fc_3: LinearBlock<B>,
    output: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv_1 = ConvBlock::new([1, 32], [3, 3], device);
        let conv_2 = ConvBlock::new([32, 64], [3, 3], device);
        let conv_3 = ConvBlock::new([64, 128], [3, 3], device);
        let conv_4 = ConvBlock::new([128, 128], [5, 5], device);
        let conv_5 = ConvBlock::new([128, 256], [5, 5], device);

        let dropout_1 = DropoutConfig::new(0.5).init();
        let fc_1 = LinearBlock::new(256 * 124 * 124, 512, device);
        let dropout_2 = DropoutConfig::new(0.5).init();
        let fc_2 = LinearBlock::new(512, 256, device);
        let dropout_3 = DropoutConfig::new(0.5).init();
        let fc_3 = LinearBlock::new(256, 128, device);
        let output = LinearConfig::new(128, 81).with_bias(false).init(device);

        return Self {
            conv_1,
            conv_2,
            conv_3,
            conv_4,
            conv_5,
            dropout_1,
            fc_1,
            dropout_2,
            fc_2,
            dropout_3,
            fc_3,
            output,
        };
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = input.dims();
        let x = input.reshape([batch_size, 1, height, width]).detach();
        let x = self.conv_1.forward(x);
        let x = self.conv_2.forward(x);
        let x = self.conv_3.forward(x);
        let x = self.conv_4.forward(x);
        let x = self.conv_5.forward(x);
        let [batch_size, channels, height, width] = x.dims();

        //linear
        let x = x.reshape([batch_size, channels * height * width]);
        let x = self.dropout_1.forward(x);
        let x = self.fc_1.forward(x);
        let x = self.dropout_2.forward(x);
        let x = self.fc_2.forward(x);
        let x = self.dropout_3.forward(x);
        let x = self.fc_3.forward(x);
        return self.output.forward(x);
    }

    pub fn forward_classification(&self, item: GoBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.board_states);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    activation: Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], device: &B::Device) -> Self {
        let conv = Conv2dConfig::new(channels, kernel_size)
            .with_stride([1, 1])
            .init(device);
        return Self {
            conv,
            activation: Relu::new(),
        };
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        return self.activation.forward(x);
    }
}

#[derive(Module, Debug)]

pub struct LinearBlock<B: Backend> {
    linear: Linear<B>,
    activation: Relu,
}

impl<B: Backend> LinearBlock<B> {
    pub fn new(input: usize, output: usize, device: &B::Device) -> Self {
        let linear = LinearConfig::new(input, output)
            .with_bias(false)
            .init(device);

        return Self {
            linear,
            activation: Relu::new(),
        };
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear.forward(input);
        let [batch_size, channels] = x.dims();
        let x = x.reshape([batch_size, channels, 1]);
        let x = x.reshape([batch_size, channels]);
        return self.activation.forward(x);
    }
}

impl<B: AutodiffBackend> TrainStep<GoBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: GoBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<GoBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: GoBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}
