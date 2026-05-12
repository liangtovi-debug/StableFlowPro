#!/usr/bin/env python3
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_SOURCE = ROOT / "Scripts" / "stableflow_probe_main.swift"
PROBE_BINARY = ROOT / ".build" / "stableflow_probe"
IMPROVED_STABILIZER = ROOT / "Sources" / "ImprovedStabilizer.swift"
GPU_STABILIZER = ROOT / "Sources" / "GPUStabilizer.swift"

SCENARIOS = [
    ("baseline", Path("/tmp/stableflow_regression2/baseline_input.mp4"), Path("/tmp/stableflow_regression2/baseline_output.mp4")),
    ("brightness", Path("/tmp/stableflow_regression2/brightness_input.mp4"), Path("/tmp/stableflow_regression2/brightness_output.mp4")),
    ("scale", Path("/tmp/stableflow_regression2/scale_input.mp4"), Path("/tmp/stableflow_regression2/scale_output.mp4")),
    ("occlusion", Path("/tmp/stableflow_regression2/occlusion_input.mp4"), Path("/tmp/stableflow_regression2/occlusion_output.mp4")),
]

TRACK_X = "360"
TRACK_Y = "133"
TRACK_W = "90"
TRACK_H = "60"
RUNS_PER_SCENARIO = 3


def run(cmd):
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return completed.stdout


def compile_probe():
    PROBE_BINARY.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "xcrun", "swiftc", "-O",
        "-o", str(PROBE_BINARY),
        str(PROBE_SOURCE),
        str(IMPROVED_STABILIZER),
        str(GPU_STABILIZER),
        "-framework", "Foundation",
        "-framework", "AVFoundation",
        "-framework", "CoreVideo",
        "-framework", "CoreImage",
        "-framework", "Metal",
        "-framework", "Vision",
        "-framework", "CoreGraphics",
    ]
    print(run(cmd), end="")


def ffprobe_dimensions(path: Path):
    output = run([
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(path),
    ])
    stream = json.loads(output)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def detect_blue_dot_error(path: Path):
    width, height = ffprobe_dimensions(path)
    center_x = width / 2.0
    center_y = height / 2.0
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    frame_size = width * height * 3
    errors = []
    while True:
        frame = proc.stdout.read(frame_size)
        if len(frame) < frame_size:
            break

        count = 0
        sum_x = 0.0
        sum_y = 0.0
        for y in range(height):
            row = y * width * 3
            for x in range(width):
                index = row + x * 3
                r = frame[index]
                g = frame[index + 1]
                b = frame[index + 2]
                if b > 200 and r < 80 and g < 80:
                    count += 1
                    sum_x += x
                    sum_y += y

        if count > 0:
            dot_x = sum_x / count
            dot_y = sum_y / count
            errors.append(math.hypot(dot_x - center_x, dot_y - center_y))

    proc.stdout.close()
    proc.wait()
    if not errors:
        return {"mean": math.nan, "max": math.nan, "frames": 0}

    return {
        "mean": sum(errors) / len(errors),
        "max": max(errors),
        "frames": len(errors),
    }


def run_probe(input_path: Path, output_path: Path):
    output = run([
        str(PROBE_BINARY),
        str(input_path),
        str(output_path),
        TRACK_X,
        TRACK_Y,
        TRACK_W,
        TRACK_H,
    ])
    analysis_line = ""
    for line in output.splitlines():
        if line.startswith("analysis "):
            analysis_line = line
            break
    return analysis_line


def run_scenario(name: str, input_path: Path, output_path: Path):
    input_metrics = detect_blue_dot_error(input_path)
    attempts = []
    for index in range(RUNS_PER_SCENARIO):
        run_output_path = output_path.with_name(f"{output_path.stem}_run{index + 1}{output_path.suffix}")
        analysis = run_probe(input_path, run_output_path)
        output_metrics = detect_blue_dot_error(run_output_path)
        attempts.append({
            "analysis": analysis,
            "output_metrics": output_metrics,
        })

    attempts.sort(key=lambda attempt: (
        math.inf if math.isnan(attempt["output_metrics"]["mean"]) else attempt["output_metrics"]["mean"],
        math.inf if math.isnan(attempt["output_metrics"]["max"]) else attempt["output_metrics"]["max"],
    ))
    chosen = attempts[len(attempts) // 2]
    median_mean = statistics.median(attempt["output_metrics"]["mean"] for attempt in attempts)
    median_max = statistics.median(attempt["output_metrics"]["max"] for attempt in attempts)
    chosen["output_metrics"]["mean"] = median_mean
    chosen["output_metrics"]["max"] = median_max
    return input_metrics, chosen["output_metrics"], chosen["analysis"]


def main():
    missing = [input_path for _, input_path, _ in SCENARIOS if not input_path.exists()]
    if missing:
        print("缺少合成回归素材：")
        for path in missing:
            print(f"  {path}")
        return 1

    compile_probe()
    print("scenario,input_mean,output_mean,input_max,output_max,visible_frames,analysis")
    for name, input_path, output_path in SCENARIOS:
        input_metrics, output_metrics, analysis = run_scenario(name, input_path, output_path)
        print(
            ",".join([
                name,
                f"{input_metrics['mean']:.3f}",
                f"{output_metrics['mean']:.3f}",
                f"{input_metrics['max']:.3f}",
                f"{output_metrics['max']:.3f}",
                str(output_metrics["frames"]),
                analysis.replace(",", ";"),
            ])
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
