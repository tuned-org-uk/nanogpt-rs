use crate::NanoChatConfig;
use crate::checkpoint::*;
use crate::gpt::GptModel;
use burn::module::Module;
use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use tempfile::TempDir;

type TestBackend = crate::backend::AutoBackend;

#[test]
fn test_save_and_load_checkpoint() {
    let device = <TestBackend as BackendTypes>::Device::default();
    let config = NanoChatConfig {
        sequence_len: 16,
        vocab_size: 64,
        n_layer: 2,
        n_head: 2,
        n_kv_head: 2,
        n_embd: 32,
        block_size: 16,
        dropout: 0.0,
    };

    let model = GptModel::<TestBackend>::new(&config, &device);

    let ids =
        burn::tensor::Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints([1, 2, 3], &device)
            .reshape([1, 3]);

    // In-memory record round-trip: isolates load_record from the file recorder,
    // and warms up WGPU kernels so the file-load comparison is stable.
    let warmup = model.forward(ids.clone(), false);
    let rec = model.clone().into_record();
    let m_in_mem = GptModel::<TestBackend>::new(&config, &device).load_record(rec);
    let l_in_mem = m_in_mem.forward(ids.clone(), false);
    let d_in_mem: Vec<f32> = (warmup.clone() - l_in_mem)
        .abs()
        .to_data()
        .to_vec()
        .unwrap();
    assert!(
        d_in_mem.iter().all(|&x| x < 1e-5),
        "in-memory roundtrip should match (max diff = {})",
        d_in_mem.iter().cloned().fold(0.0f32, f32::max)
    );

    // File round-trip
    let temp_dir = TempDir::new().unwrap();
    save_checkpoint(&model, &config, temp_dir.path()).unwrap();

    let (loaded_model, loaded_config) =
        load_checkpoint::<TestBackend>(temp_dir.path(), &device).unwrap();

    assert_eq!(config.vocab_size, loaded_config.vocab_size);
    assert_eq!(config.n_layer, loaded_config.n_layer);

    let logits1 = model.forward(ids.clone(), false);
    let logits2 = loaded_model.forward(ids, false);

    let diff: Vec<f32> = (logits1.clone() - logits2.clone())
        .abs()
        .to_data()
        .to_vec()
        .unwrap();

    let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "Loaded model weights should match original (max diff = {})",
        max_diff
    );
}
