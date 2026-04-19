# Data Merge

`data_merge/` is a standalone project for turning three panorama data sources into one SFT-ready corpus.

It normalizes:

- `results_final_v2`: canonical samples only. The merger recursively reads `scene_id/viewpoint_id/canonical_samples.jsonl`, and for each sample derives the ERP image path as `Realsee3D/real_world_data/{scene_id}/viewpoints/{viewpoint_id}/panoImage_1600.jpg`.
- `output_v.jsonl`: long-form scene captions. Each panorama is expanded into multiple SFT samples, including a full caption sample plus section-level samples such as `GLOBAL LAYOUT` and `FINAL RECONSTRUCTION`.
- `pano_grounding_train_factory.json`: existing grounding SFT items. These are kept in the same chat-style format while remapping dead path prefixes.

## Training formats

The builder supports two final training formats, selected by `--training-format`.

### 1. `simple`

This writes `training_data.json`. Each item looks like:

```json
{
  "images": ["/workspace/data_dir/data_user/public_data/360video/Realsee3D/real_world_data/.../panoImage_1600.jpg"],
  "messages": [
    {"role": "user", "content": "<image>Describe the global layout of this 360 panorama."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 2. `multimodal_blocks`

This writes `training_data_multimodal_blocks.json`. It uses a multimodal chat format with typed content blocks:

```json
{
  "id": "scene_00001:grounding:0022",
  "images": ["/workspace/data_dir/data_user/public_data/360video/Realsee3D/real_world_data/.../panoImage_1600.jpg"],
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are a multimodal assistant specialized in ERP ..."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Return only the BFOV [yaw, pitch, x_fov, y_fov] of curved off-white fabric sofa."},
        {"type": "image"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "bfov=[-152.2, 33.1, 66.1, 55.6]"}
      ]
    }
  ]
}
```

This is a multimodal chat SFT format where each message can contain typed blocks such as `text` and `image`. It is commonly used by VLM instruction-tuning pipelines and OpenAI-style multimodal chat data processors.

For `multimodal_blocks`, the system prompt is automatically injected as:

```text
You are a multimodal assistant specialized in ERP (equirectangular projection) panoramic image understanding.

The input image is a 360-degree panorama represented in ERP format. You must understand it as a continuous surrounding scene rather than a normal perspective image. The horizontal dimension wraps around, so the left and right boundaries may correspond to adjacent directions in real space.

Your job is to understand the panoramic scene faithfully and complete image-grounded tasks such as recognition, captioning, attribute description, existence checking, counting, grounding, scene understanding, directional reasoning, depth comparison, and 3D spatial relation reasoning.
```

The builder still also writes normalized debug files with metadata:

- `normalized_results_final_v2.jsonl`
- `normalized_caption_sft.jsonl`
- `normalized_grounding_sft.jsonl`
- `merged_sft.jsonl`

## Path remapping

The merger rewrites these unavailable prefixes:

- `/workspace/data_dir/USB_data2/wcp_pano_training/vqa_generation/real_world_data`
  -> `/workspace/data_dir/data_user/public_data/360video/Realsee3D/real_world_data`
- `/workspace/data_dir/USB_data/wcp_data/ReplicaPano`
  -> `/workspace/data_dir/data_user/public_data/360video/ReplicaPano`

## Run with bundled mock data

```bash
python3 /Users/wcp/code/erp_data_pipeline/data_merge/scripts/build_sft_dataset.py \
  --use-mock-data \
  --output-dir /Users/wcp/code/erp_data_pipeline/data_merge/examples/mock_output
```

That mock bundle includes:

- 2 `canonical_samples.jsonl` records under `results_final_v2/scene_00001/1753781394/`
- 1 caption record that is split into 7 SFT samples
- 1 grounding record

## Run against server data

```bash
python3 /Users/wcp/code/erp_data_pipeline/data_merge/scripts/build_sft_dataset.py \
  --results-dir /workspace/data_dir/data_user/wcp/world-model/generation_pano_base/qa_generate/results_final_v2 \
  --caption-jsonl /workspace/data_dir/USB_data2/wcp_pano_training/pano_data/vqa_generation/output_v.jsonl \
  --grounding-json /workspace/data_dir/USB_data/wcp_data/ReplicaPano/grounding/pano_grounding_train_factory.json \
  --training-format multimodal_blocks \
  --workers 16 \
  --output-dir /workspace/data_dir/data_user/wcp/data_merge_outputs/sft_merge_v1
```

`--workers` controls thread-level parallelism for:

- scanning and parsing many `canonical_samples.jsonl` files
- splitting caption records
- normalizing grounding records

Main files written to the output directory:

- `training_data.json` for `--training-format simple`
- `training_data_multimodal_blocks.json` for `--training-format multimodal_blocks`
- `normalized_results_final_v2.jsonl`
- `normalized_caption_sft.jsonl`
- `normalized_grounding_sft.jsonl`
- `merged_sft.jsonl`
- `stats.json`

## Tests

```bash
python3 -m unittest discover -s /Users/wcp/code/erp_data_pipeline/data_merge/tests
```

## Video Extraction

To uniformly sample ERP video frames from the two outdoor video sources:

```bash
python3 /Users/wcp/code/erp_data_pipeline/data_merge/scripts/extract_uniform_video_frames.py \
  --sphere360-dir /workspace/data_dir/data_user/public_data/360video/Sphere360/videos \
  --panowan-dir /workspace/data_dir/data_user/public_data/360video/panowan \
  --output-images-root /workspace/data_dir/data_user/public_data/360video/outdoor/images \
  --output-manifest /workspace/data_dir/data_user/public_data/360video/outdoor/frame_manifest.json \
  --frames-per-video 5 \
  --resize-width 2048 \
  --resize-height 1024 \
  --workers 16
```

Behavior:

- Samples 5 frames per video at uniform time positions along the timeline.
- Resizes every extracted frame directly to `2048x1024`.
- Saves frames under `outdoor/images/{dataset}/{relative_video_path_without_suffix}/frame_00.jpg`.
- Writes one final JSON manifest at `frame_manifest.json`.
- If the 5 expected frames already exist, the script reuses them by default so interrupted runs can resume safely.
- Add `--overwrite` to force re-extraction.

The manifest structure is a simple JSON list:

```json
[
  {
    "image_path": "/workspace/.../outdoor/images/Sphere360/subdir/demo/frame_00.jpg",
    "source": "/workspace/.../Sphere360/videos/subdir/demo.mp4",
    "scene_id": "demo",
    "viewpoint_id": "frame_00"
  }
]
```

## Benchmark Image Manifest

If you already have a benchmark image directory copied to:

- `/workspace/data_dir/data_user/public_data/360video/outdoor/test`

you can build a manifest JSON for those images with:

```bash
python3 /Users/wcp/code/erp_data_pipeline/data_merge/scripts/build_image_manifest.py \
  --dataset-root /workspace/data_dir/data_user/public_data/360video/outdoor/test \
  --output-manifest /workspace/data_dir/data_user/public_data/360video/outdoor/test/image_manifest.json \
  --dataset-name real_360_test \
  --resize-width 2048 \
  --resize-height 1024 \
  --progress-every 200 \
  --workers 16
```

This script:

- auto-detects `images/` under the dataset root if present
- scans all common image files
- resizes every image in place to `2048x1024`
- tries to match each image filename back to either `search_results.jsonl` or `metadata.jsonl`
- prints periodic progress while resizing and indexing large image sets
- skips unreadable or broken images instead of aborting the whole run
- writes a simple JSON list with one record per image

For datasets like `panox/`, the matcher will also read `metadata.jsonl` rows such as:

- `image_name`
- `local_path`
- `asset_url`
- `id`

and use them to align images with their source metadata while keeping the final manifest format unchanged.

Example output structure:

```json
[
  {
    "image_path": "/workspace/.../images/commons__File_Demo.jpg",
    "source": "https://commons.wikimedia.org/wiki/File:Demo.jpg",
    "scene_id": "commons__File_Demo",
    "viewpoint_id": "commons__File_Demo"
  }
]
```

## External Benchmark Training Sets

You can also collect three external 360-degree training corpora and export them in the same `multimodal_blocks` chat format:

- `osr_bench_train_multimodal_blocks.json`
- `osr_bench_validation_multimodal_blocks.json`
- `osr_bench_test_multimodal_blocks.json`
- `thinking_in_360_training_multimodal_blocks.json`
- `panoenv_training_multimodal_blocks.json`

Run:

```bash
python3 /Users/wcp/code/erp_data_pipeline/data_merge/scripts/build_external_benchmark_sft.py \
  --output-dir /workspace/data_dir/data_user/wcp/data_merge_outputs/external_benchmark_sft \
  --cache-dir /workspace/data_dir/data_user/wcp/data_merge_outputs/external_benchmark_cache
```

Default behavior:

- `thinking_in_360_training_multimodal_blocks.json`: exported by default from the official panorama SFT releases.
- `panoenv_training_multimodal_blocks.json`: exported by default, because PanoEnv has an official train split and is panorama-native.
- `osr_bench_train_multimodal_blocks.json`: written by explicit opt-in only.
- `osr_bench_validation_multimodal_blocks.json`: written by explicit opt-in only.
- `osr_bench_test_multimodal_blocks.json`: written by explicit opt-in only.

The OSR-Bench files are empty by default on purpose:

- OSR-Bench is mainly a benchmark release. The public package exposes images plus one `qa.csv`, but no official QA train split. If you force-convert it for training, benchmark evaluation on OSR-Bench is no longer clean.

The converter uses official sources only:

- OSR-Bench: official `qa.csv` plus the benchmark images from [UUUserna/OSR-Bench](https://huggingface.co/datasets/UUUserna/OSR-Bench)
- Thinking in 360: the official panorama SFT releases [humanoid-vstar/hos_sft_panorama](https://huggingface.co/datasets/humanoid-vstar/hos_sft_panorama) and [humanoid-vstar/hps_sft_panorama](https://huggingface.co/datasets/humanoid-vstar/hps_sft_panorama)
- PanoEnv: the official train split loaded through [7zkk/PanoEnv](https://huggingface.co/datasets/7zkk/PanoEnv) or [guangmulizi/PanoEnv](https://huggingface.co/datasets/guangmulizi/PanoEnv)

Important training notes:

- OSR-Bench can be force-converted into training data, but the public release does not expose a clean QA train split. If you enable it, the exporter creates deterministic `9:1:1` train/validation/test splits at the image-source level so all questions from the same panorama stay in the same split.
- Thinking in 360 now uses the official panorama SFT releases rather than the older perspective-view SFT dumps.
- PanoEnv uses only the official `train` split and, by default, keeps only `open_ended` questions as a higher-quality subset.

The script writes a companion `external_benchmark_stats.json` file with these notes and per-file counts.

Useful options:

- `--max-osr-records N`
- `--max-thinking-records N`
- `--max-panoenv-records N`
- `--panoenv-question-types open_ended`
- `--include-osr-bench`
- `--skip-thinking-in-360`
- `--skip-panoenv`

Example smoke test:

```bash
python3 /Users/wcp/code/erp_data_pipeline/data_merge/scripts/build_external_benchmark_sft.py \
  --output-dir /Users/wcp/code/erp_data_pipeline/data_merge/examples/external_benchmark_output \
  --cache-dir /Users/wcp/code/erp_data_pipeline/data_merge/examples/external_benchmark_cache \
  --max-osr-records 2 \
  --include-osr-bench \
  --skip-panoenv
```

Notes:

- Thinking in 360 downloads official panorama zip archives, then recursively finds the bundled JSON/JSONL manifest inside the extracted package.
- PanoEnv conversion relies on the `datasets` package because the official train split is loaded through the dataset loader rather than a plain JSONL file. If needed, install it with `pip install datasets pillow`.
- `PanoEnv` official train size is `415` panorama images and `10,340` QA pairs according to the dataset README. The default `open_ended` filter keeps a cleaner subset for SFT.
- The script always writes stable output files, even when one source is intentionally skipped.
- The exported format is the same multimodal training schema used elsewhere in this repo:
  - `id`
  - `messages`
  - `images`
