# vapoursynth-cambi

[Contrast Aware Multiscale Banding Index][CAMBI] (CAMBI) as a [VapourSynth][]
plugin. This is a pure Rust+VapourSynth implementation of Netflix's banding
detection algorithm without a dependency on libvmaf.

## Install

Windows: Download a release from the [Releases][] page and unzip `cambi.dll`
into a [plugins directory][plugin-autoloading]. There are separate artifacts for
Raptor Lake (`*-raptorlake.zip`) and AMD Zen 4 (`*-znver4.zip`) CPUs which may
or may not have better performance than the plain x86_64 build.

## API

```python
cambi.Cambi(
    clip: vs.VideoNode,
    window_size: int = 65,
    topk: float = 0.6,
    tvi_threshold: float = 0.019,
    max_log_contrast: int = 2,
    prop: str = "CAMBI",
) -> vs.VideoNode
```

Calculates the CAMBI score of each frame of `clip` and stores it in the frame
property named `prop`.

- `clip` — Constant format 10-bit integer clip.
- `window_size` — Window size to compute CAMBI. Note that libvmaf's documented
  default is 63 but internally it actually uses 65, so this plugin defaults to
  65 as well.
- `topk` — Ratio of pixels for the spatial pooling computation.
- `tvi_threshold` — Visibility threshold for luminance ΔL < tvi_threshold\*L_mean for BT.1886.
- `max_log_contrast` — Maximum contrast in log luma level (2^max_log_contrast)
  at 10-bit. Default 2 is equivalent to 4 luma levels at 10-bit.
- `prop` — Name of the frame property to store the CAMBI score in.

## Benchmark

With warm caches, this plugin is about 2.7 times faster than `akarin.Cambi()` on
1080p clips (222.3 fps vs. 80.5 fps over 10 runs of 1001 frames each).

```python
from vstools import core, initialize_clip, set_output, vs

core.set_affinity(16)

src = core.bs.VideoSource(r"/path/to/1080p_video.m2ts")
src = initialize_clip(src, bits=10)

set_output(core.cambi.Cambi(src, window_size=65))
set_output(core.akarin.Cambi(src, window_size=65))
```

```bash
$ hyperfine --warmup 1 'vspipe test.py --end 1000 -o {output_node} .' -P output_node 0 1
Benchmark 1: vspipe test.py --end 1000 -o 0 .
  Time (mean ± σ):      4.502 s ±  0.043 s    [User: 53.877 s, System: 4.763 s]
  Range (min … max):    4.450 s …  4.575 s    10 runs

Benchmark 2: vspipe test.py --end 1000 -o 1 .
  Time (mean ± σ):     12.435 s ±  0.080 s    [User: 183.298 s, System: 2.283 s]
  Range (min … max):   12.342 s … 12.570 s    10 runs

Summary
  vspipe test.py --end 1000 -o 0 . ran
    2.76 ± 0.03 times faster than vspipe test.py --end 1000 -o 1 .
```

## Build

Rust v1.81.0-nightly and cargo may be used to build the project. Older versions
will likely work fine but they aren't explicitly supported.

```bash
$ git clone https://github.com/sgt0/vapoursynth-cambi.git
$ cd vapoursynth-cambi

# Debug build.
$ cargo build

# Release (optimized) build.
$ cargo build --release

# Release build optimized for the host CPU.
$ RUSTFLAGS="-C target-cpu=native" cargo build --release
```

[CAMBI]: https://github.com/Netflix/vmaf/blob/v3.0.0/resource/doc/papers/CAMBI_PCS2021.pdf
[VapourSynth]: https://www.vapoursynth.com
[Releases]: https://github.com/sgt0/vapoursynth-cambi/releases
[plugin-autoloading]: https://www.vapoursynth.com/doc/installation.html#plugin-autoloading
