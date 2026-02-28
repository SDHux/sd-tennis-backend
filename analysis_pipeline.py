"""
SD Tennis Lessons — Video Analysis Pipeline
Handles: video upload → frame extraction → Claude API → structured report

Dependencies:
    pip install anthropic opencv-python-headless Pillow requests

FFmpeg must be installed on the server:
    macOS: brew install ffmpeg
    Ubuntu: apt install ffmpeg
    Windows: https://ffmpeg.org/download.html
"""

import anthropic
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from coaching_prompt import SYSTEM_PROMPT, build_user_prompt


# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

# How many frames to extract per video
# 8-12 is the sweet spot: enough for sequence analysis, not too many tokens
FRAMES_TO_EXTRACT = 10

# Max video duration to process (seconds) — keeps costs predictable
MAX_VIDEO_DURATION = 120  # 2 minutes

# Claude model to use — Opus gives the best coaching analysis
CLAUDE_MODEL = "claude-opus-4-6"


# ─────────────────────────────────────────────────────────
# FRAME EXTRACTION
# ─────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int = FRAMES_TO_EXTRACT) -> list[str]:
    """
    Extracts evenly-spaced frames from a video using FFmpeg.
    Returns list of base64-encoded JPEG strings.
    
    Why FFmpeg over OpenCV?
    - More reliable with diverse video formats (MOV, MP4, HEVC from iPhones)
    - Better handles variable frame rates from phone cameras
    - Faster for extraction-only tasks
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get video duration first
    duration = get_video_duration(str(video_path))
    if duration > MAX_VIDEO_DURATION:
        print(f"Warning: Video is {duration:.0f}s. Analyzing first {MAX_VIDEO_DURATION}s only.")
        duration = MAX_VIDEO_DURATION

    # Calculate frame extraction interval
    interval = duration / num_frames

    frames_b64 = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract frames at calculated intervals
        # -vf fps=1/{interval} = extract one frame every N seconds
        # -q:v 2 = high quality JPEG (1=best, 31=worst)
        # -vframes {num_frames} = stop after N frames
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps=1/{interval:.2f}",
            "-q:v", "2",
            "-vframes", str(num_frames),
            "-y",  # overwrite without asking
            f"{tmpdir}/frame_%03d.jpg"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        # Load and encode frames
        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))

        if not frame_files:
            raise RuntimeError("No frames were extracted. Check video format.")

        for frame_path in frame_files:
            with open(frame_path, "rb") as f:
                frame_data = base64.standard_b64encode(f.read()).decode("utf-8")
                frames_b64.append(frame_data)

    print(f"Extracted {len(frames_b64)} frames from {video_path.name}")
    return frames_b64


def get_video_duration(video_path: str) -> float:
    """Uses FFprobe to get video duration in seconds."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


# ─────────────────────────────────────────────────────────
# CLAUDE API CALL
# ─────────────────────────────────────────────────────────

def analyze_frames_with_claude(
    frames_b64: list[str],
    stroke_type: str = "general",
    student_info: Optional[dict] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Sends extracted frames to Claude with the coaching system prompt.
    Returns the full analysis as a formatted string.
    
    Args:
        frames_b64: List of base64-encoded JPEG frames
        stroke_type: One of: forehand, backhand_one_handed, backhand_two_handed, 
                     serve, volley, general
        student_info: Dict with optional keys: name, level, age, concerns
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
    
    Returns:
        Formatted coaching analysis string
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    # Build the message content — images first, then the text prompt
    content = []

    # Add each frame as an image block
    for i, frame_b64 in enumerate(frames_b64):
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame_b64
            }
        })
        # Add a small label so Claude can reference frame numbers
        content.append({
            "type": "text",
            "text": f"[Frame {i + 1}]"
        })

    # Add the analysis request
    user_prompt = build_user_prompt(
        frames_count=len(frames_b64),
        stroke_type=stroke_type,
        student_info=student_info
    )
    content.append({
        "type": "text",
        "text": user_prompt
    })

    print(f"Sending {len(frames_b64)} frames to Claude for analysis...")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return response.content[0].text


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE — this is what your web server calls
# ─────────────────────────────────────────────────────────

def analyze_tennis_video(
    video_path: str,
    stroke_type: str = "general",
    student_info: Optional[dict] = None,
    api_key: Optional[str] = None
) -> dict:
    """
    Full pipeline: video file → coaching analysis report.
    
    This is the single function your web backend calls.
    
    Args:
        video_path: Path to uploaded video file
        stroke_type: Type of stroke to analyze
        student_info: Optional student context
        api_key: Anthropic API key
    
    Returns:
        Dict with keys:
            - analysis: Full coaching analysis text
            - frames_analyzed: Number of frames processed
            - stroke_type: Stroke type analyzed
            - student_name: Student name if provided
            - success: Boolean
            - error: Error message if success is False
    
    Example usage:
        result = analyze_tennis_video(
            video_path="/uploads/alex_forehand.mp4",
            stroke_type="forehand",
            student_info={
                "name": "Alex",
                "level": "intermediate",
                "age": 16,
                "concerns": "Ball keeps going into the net under pressure"
            }
        )
        print(result["analysis"])
    """
    try:
        # Step 1: Extract frames
        frames = extract_frames(video_path)

        # Step 2: Send to Claude
        analysis = analyze_frames_with_claude(
            frames_b64=frames,
            stroke_type=stroke_type,
            student_info=student_info,
            api_key=api_key
        )

        return {
            "success": True,
            "analysis": analysis,
            "frames_analyzed": len(frames),
            "stroke_type": stroke_type,
            "student_name": student_info.get("name", "Student") if student_info else "Student",
            "error": None
        }

    except FileNotFoundError as e:
        return {"success": False, "error": f"Video file not found: {e}", "analysis": None}
    except RuntimeError as e:
        return {"success": False, "error": f"Processing error: {e}", "analysis": None}
    except anthropic.APIError as e:
        return {"success": False, "error": f"Claude API error: {e}", "analysis": None}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}", "analysis": None}


# ─────────────────────────────────────────────────────────
# COST ESTIMATOR — handy for business planning
# ─────────────────────────────────────────────────────────

def estimate_cost_per_analysis(num_frames: int = FRAMES_TO_EXTRACT) -> dict:
    """
    Rough cost estimate per student submission.
    Based on Anthropic's current pricing for Claude Opus.
    
    Images are ~1,600 tokens each at standard resolution.
    System prompt ~800 tokens. Response ~500 tokens.
    """
    # Approximate token counts
    image_tokens = num_frames * 1600
    system_tokens = 800
    response_tokens = 500
    total_input_tokens = image_tokens + system_tokens
    
    # Opus pricing (as of early 2026)
    input_cost_per_million = 15.00   # $15 per 1M input tokens
    output_cost_per_million = 75.00  # $75 per 1M output tokens
    
    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (response_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost
    
    return {
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": response_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "cost_at_25_per_session": f"${25 - total_cost:.2f} gross margin per submission"
    }


if __name__ == "__main__":
    # Test the cost estimator
    cost = estimate_cost_per_analysis()
    print("=== COST ESTIMATE PER ANALYSIS ===")
    for k, v in cost.items():
        print(f"  {k}: {v}")
    
    # Uncomment to test with a real video:
    # result = analyze_tennis_video(
    #     video_path="test_video.mp4",
    #     stroke_type="forehand",
    #     student_info={"name": "Test Student", "level": "beginner"}
    # )
    # print(result["analysis"])
