from __future__ import annotations

from collections import OrderedDict


DEFAULT_RESULTS_DIR = "/workspace/data_dir/data_user/wcp/world-model/generation_pano_base/qa_generate/results_final_v2"
DEFAULT_CAPTION_JSONL = "/workspace/data_dir/USB_data2/wcp_pano_training/pano_data/vqa_generation/output_v.jsonl"
DEFAULT_GROUNDING_JSON = "/workspace/data_dir/USB_data/wcp_data/ReplicaPano/grounding/pano_grounding_train_factory.json"
REALSEE_PUBLIC_ROOT = "/workspace/data_dir/data_user/public_data/360video/Realsee3D/real_world_data"

PATH_REPLACEMENTS = [
    (
        "/workspace/data_dir/USB_data2/wcp_pano_training/vqa_generation/real_world_data",
        "/workspace/data_dir/data_user/public_data/360video/Realsee3D/real_world_data",
    ),
    (
        "/workspace/data_dir/USB_data/wcp_data/ReplicaPano",
        "/workspace/data_dir/data_user/public_data/360video/ReplicaPano",
    ),
]

CAPTION_SECTION_PROMPTS = OrderedDict(
    [
        ("GLOBAL LAYOUT", "Describe the global layout of this 360 panorama."),
        (
            "TOPOLOGICAL RELATIONS",
            "Describe the topological relations in this panorama, including the front, right, back, and left areas.",
        ),
        ("NAVIGABLE FRONTIERS", "What are the navigable frontiers and movement paths in this scene?"),
        ("PHYSICAL CONSTRAINTS", "What physical constraints or barriers are present in this scene?"),
        ("SPATIAL REASONING SUMMARY", "Summarize the key spatial reasoning cues in this panorama."),
        ("FINAL RECONSTRUCTION", "Give the final reconstruction of this scene."),
    ]
)

FULL_CAPTION_PROMPT = "Provide a detailed full-scene caption for this 360 panorama."

ERP_MULTIMODAL_SYSTEM_PROMPT = """You are a multimodal assistant specialized in ERP (equirectangular projection) panoramic image understanding.

The input image is a 360-degree panorama represented in ERP format. You must understand it as a continuous surrounding scene rather than a normal perspective image. The horizontal dimension wraps around, so the left and right boundaries may correspond to adjacent directions in real space.

Your job is to understand the panoramic scene faithfully and complete image-grounded tasks such as recognition, captioning, attribute description, existence checking, counting, grounding, scene understanding, directional reasoning, depth comparison, and 3D spatial relation reasoning."""
