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
