"""Simple JSONL video dataset for Kairos training.

A manifest contains one local video and one caption per line. Videos are decoded
lazily in the DataLoader worker, temporally sampled and spatially resized to the
configured target shape, then returned as the list-of-PIL-frames contract
expected by the Kairos pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import imageio.v3 as iio
from PIL import Image
from torch.utils.data import Dataset


class SimpleVideoDataset(Dataset):
    """Read local videos from a JSONL manifest.

    Parameters
    ----------
    manifest:
        JSONL file.  Each non-empty line must contain ``video`` and either
        ``caption`` or ``prompt``.  Paths are relative to ``data_root``.
    data_root:
        Root directory for the manifest paths. Paths must resolve within this
        directory.
    width, height, num_frames, fps:
        Target output width, height, frame count, and sampling frame rate. The
        defaults match the 832x480/81-frame Kairos training recipe. Source
        videos may have different resolution, frame count, and frame rate.
    """

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path = ".",
        *,
        width: int = 832,
        height: int = 480,
        num_frames: int = 81,
        fps: float = 16.0,
    ) -> None:
        super().__init__()
        self.manifest = Path(manifest).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.width = int(width)
        self.height = int(height)
        self.num_frames = int(num_frames)
        self.target_fps = float(fps)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.num_frames <= 0 or self.num_frames % 4 != 1:
            raise ValueError("num_frames must be positive and satisfy num_frames % 4 == 1")
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("fps must be positive")
        if not self.manifest.is_file():
            raise FileNotFoundError(f"manifest does not exist: {self.manifest}")
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"data_root does not exist: {self.data_root}")

        self.records: List[Dict[str, Any]] = []
        with self.manifest.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {self.manifest}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"manifest line {line_number} must be an object")
                video = record.get("video")
                caption = record.get("caption", record.get("prompt"))
                if not isinstance(video, str) or not video.strip():
                    raise ValueError(f"manifest line {line_number} has no non-empty 'video'")
                if not isinstance(caption, str):
                    raise ValueError(
                        f"manifest line {line_number} must contain string 'caption' (or 'prompt')"
                    )
                # An optional id is useful for logging but is not passed to the model.
                self.records.append({"video": video, "caption": caption, "id": record.get("id")})

        if not self.records:
            raise ValueError(f"manifest is empty: {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_video_path(self, value: str) -> Path:
        candidate_text = value.strip()
        if "://" in candidate_text:
            raise ValueError(f"s3 path is not allowed: {value!r}")
        candidate = (self.data_root / candidate_text).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"video does not exist: {candidate}")
        return candidate

    def _decode_video(self, path: Path):
        try:
            with iio.imopen(uri=str(path), io_mode="r", plugin="pyav") as reader:
                metadata = reader.metadata()
        except ImportError as exc:  # pragma: no cover - exercised in environment setup
            raise ImportError(
                "ImageIO's PyAV plugin is required for SimpleVideoDataset; "
                "install requirements-train.txt"
            ) from exc

        # ``fps`` is the target sampling rate, not a source-video constraint.
        # This follows A's LoadVideo_IIO3 behavior: target timestamps are
        # mapped to source frame indices. Repeated indices are intentional when
        # the source FPS is lower than the target FPS.
        raw_source_fps = metadata.get("fps", 30.0)
        source_fps = float(raw_source_fps)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"video has no usable FPS metadata: {path}")
        frame_ids = [
            int(round(frame_index * source_fps / self.target_fps))
            for frame_index in range(self.num_frames)
        ]
        positions_by_frame_id: Dict[int, List[int]] = {}
        for output_index, source_frame_id in enumerate(frame_ids):
            positions_by_frame_id.setdefault(source_frame_id, []).append(output_index)

        frames: List[Image.Image | None] = [None] * self.num_frames
        max_source_frame_id = frame_ids[-1]
        try:
            for source_frame_id, frame in enumerate(
                iio.imiter(uri=str(path), plugin="pyav")
            ):
                if source_frame_id > max_source_frame_id:
                    break
                output_indices = positions_by_frame_id.get(source_frame_id)
                if output_indices is None:
                    continue

                image = Image.fromarray(frame).convert("RGB")
                # Match A's Video_Resize_To_Target_WH: direct resize to the
                # requested dimensions, without aspect-ratio preservation or
                # center cropping.
                image = image.resize(
                    (self.width, self.height),
                    resample=Image.Resampling.BILINEAR,
                )
                for output_index in output_indices:
                    frames[output_index] = image

            if any(frame is None for frame in frames):
                last_available = max(
                    (index for index, frame in enumerate(frames) if frame is not None),
                    default=-1,
                )
                raise ValueError(
                    f"{path} is too short for {self.num_frames} frames at "
                    f"target fps {self.target_fps:g}; "
                    f"decoded target frame positions through {last_available}"
                )
            # The check above guarantees that no Optional values remain.
            return [frame for frame in frames if frame is not None]
        except ImportError as exc:  # pragma: no cover - exercised in environment setup
            raise ImportError(
                "ImageIO's PyAV plugin is required for SimpleVideoDataset; "
                "install requirements-train.txt"
            ) from exc

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        path = self._resolve_video_path(record["video"])
        return {
            "video": self._decode_video(path),
            "prompt": record["caption"],
        }


def collate_video_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Keep PIL frames as nested lists for ``KairosEmbodiedWAMPipeline``."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    return {
        "video": [item["video"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
    }


__all__ = ["SimpleVideoDataset", "collate_video_batch"]
