"""Prepares a raw phone video for the pipeline: strips location, device and timestamp metadata and the audio, fixes HDR colours, and shrinks it to 1280 px on the long side (usage: python pipeline/clean_video.py SRC DST [SRC DST ...])."""

import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MAX_SIDE = 1280
TONEMAP = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv"
SCALE = f"scale='if(gt(iw,ih),min({MAX_SIDE},iw),-2)':'if(gt(iw,ih),-2,min({MAX_SIDE},ih))'"


def is_hdr(src):
    """Return True if the video is HDR, which has to be converted or its colours look washed out."""
    info = subprocess.run([FFMPEG, "-hide_banner", "-i", str(src)], capture_output=True, text=True).stderr
    return bool(re.search(r"Stream #0:\d+.*Video:.*(arib-std-b67|smpte2084)", info))


def clean_video(src, dst):
    """Save a cleaned copy of src at dst: no metadata or audio, normal (SDR) colours, long side at most MAX_SIDE, same frame rate."""
    filters = [TONEMAP] if is_hdr(src) else []
    filters += [SCALE, "format=yuv420p"]
    command = [
        FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-map", "0:v:0", "-map_metadata", "-1", "-map_chapters", "-1",
        "-an", "-sn", "-dn", "-vf", ",".join(filters), "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-fflags", "+bitexact", "-flags:v", "+bitexact",
        "-metadata:s:v:0", "handler_name=", "-movflags", "+faststart", str(dst),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    paths = sys.argv[1:]
    for src, dst in zip(paths[::2], paths[1::2]):
        clean_video(Path(src), Path(dst))
        print(f"{src} -> {dst}")
