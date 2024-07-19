#![deny(clippy::all, clippy::pedantic, clippy::nursery, clippy::perf)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unreadable_literal)]

mod cambi;
mod luminance;

use std::ffi::{c_void, CStr, CString};

use cambi::{adjust_window_size, cambi_score, get_tvi_for_diff, CambiParams, ContrastArrays};
use const_str::cstr;
use luminance::{Bt1886, LumaRange};
use vapoursynth4_rs::{
  core::CoreRef,
  declare_plugin,
  ffi::VSColorRange,
  frame::{Frame, FrameContext, VideoFrame},
  key,
  map::{AppendMode, KeyStr, MapMut, MapRef, Value},
  node::{ActivationReason, Dependencies, Filter, FilterDependency, Node, RequestPattern, VideoNode},
  utils::is_constant_video_format,
  SampleType,
};

/// Contrast Aware Multiscale Banding Index (CAMBI) as a VapourSynth plugin.
#[allow(clippy::doc_markdown)]
struct CambiFilter {
  node: VideoNode,

  /// CAMBI parameters that are constant across frames.
  cambi_params: CambiParams,
  tvi_threshold: f64,

  /// Name of the frame property to store the CAMBI score in.
  prop: CString,
}

impl Filter for CambiFilter {
  type Error = &'static CStr;
  type FrameType = VideoFrame;
  type FilterData = ();

  fn create(
    input: MapRef<'_>,
    output: MapMut<'_>,
    _data: Option<Box<Self::FilterData>>,
    mut core: CoreRef,
  ) -> Result<(), Self::Error> {
    let Ok(node) = input.get_video_node(key!("clip"), 0) else {
      return Err(cstr!("cambi: failed to get clip."));
    };

    let n = node.clone();
    let vi = n.info();

    if !is_constant_video_format(vi) || vi.format.sample_type != SampleType::Integer || vi.format.bits_per_sample != 10
    {
      return Err(cstr!("cambi: only constant format 10-bit integer input is supported."));
    }

    // CAMBI parameters.
    let window_size = adjust_window_size(
      input.get_int(key!("window_size"), 0).unwrap_or(65) as u16,
      vi.width,
      vi.height,
    );
    let topk = input.get_float(key!("window_size"), 0).unwrap_or(0.6) as f32;
    let tvi_threshold = input.get_float(key!("tvi_threshold "), 0).unwrap_or(0.019);
    let max_log_contrast = input.get_int(key!("max_log_contrast "), 0).unwrap_or(2);
    let num_diffs: i32 = 1 << max_log_contrast;
    let contrast_arrays = ContrastArrays::new(num_diffs);
    let mut filter = Self {
      node,
      tvi_threshold,
      cambi_params: CambiParams {
        contrast_arrays,
        num_diffs,
        topk,
        window_size,
        width: vi.width,
        height: vi.height,
      },
      prop: CString::new(input.get_utf8(key!("prop"), 0).unwrap_or("CAMBI"))
        .expect("cambi: should be able to create a C-compatible prop name."),
    };

    let deps = [FilterDependency {
      source: filter.node.as_mut_ptr(),
      request_pattern: RequestPattern::StrictSpatial,
    }];

    core.create_video_filter(
      output,
      Self::NAME,
      vi,
      Box::new(filter),
      Dependencies::new(&deps).unwrap(),
    );

    Ok(())
  }

  fn get_frame(
    &self,
    n: i32,
    activation_reason: ActivationReason,
    _frame_data: *mut *mut c_void,
    mut ctx: FrameContext,
    core: CoreRef,
  ) -> Result<Option<VideoFrame>, Self::Error> {
    match activation_reason {
      ActivationReason::Initial => {
        ctx.request_frame_filter(n, &self.node);
      }
      ActivationReason::AllFramesReady => {
        let src = self.node.get_frame_filter(n, &mut ctx);
        let format = src.get_video_format();

        // All of this `tvi_for_diff` initialization could be moved to filter
        // initialization at the cost of assuming constant color range and
        // transfer. Benchmarks indicate that this is would be a <1% performance
        // improvement, which is probably not worth the added limitation.
        let props = src.properties().expect("cambi: should be able to get frame props.");
        let range = match props
          .get_int_saturated(key!("_ColorRange"), 0)
          .unwrap_or(VSColorRange::VSC_RANGE_LIMITED as i32)
        {
          0 => VSColorRange::VSC_RANGE_FULL,
          _ => VSColorRange::VSC_RANGE_LIMITED, // TODO: throw on unknown color range
        };
        let mut tvi_for_diff = Vec::<u16>::with_capacity(self.cambi_params.num_diffs as usize);
        for d in 0..self.cambi_params.num_diffs {
          tvi_for_diff.push(
            (get_tvi_for_diff(
              i32::from(self.cambi_params.contrast_arrays.diffs_to_consider[d as usize]),
              self.tvi_threshold,
              format.bits_per_sample,
              &LumaRange::new(format.bits_per_sample, range),
              &Bt1886, // TODO: remove hardcoded BT.1886.
            ) + self.cambi_params.num_diffs) as u16,
          );
        }

        let score = cambi_score(
          &mut core.copy_frame(&src),
          &mut core.copy_frame(&src),
          &tvi_for_diff,
          &self.cambi_params,
        );

        let mut dst = core.copy_frame(&src);
        let mut props = dst.properties_mut().expect("cambi: should be able to get frame props.");
        props
          .set(KeyStr::from_cstr(&self.prop), Value::Float(score), AppendMode::Replace)
          .expect("cambi: should be able to set frame prop.");
        return Ok(Some(dst));
      }
      ActivationReason::Error => {}
    }

    Ok(None)
  }

  const NAME: &'static CStr = cstr!("Cambi");
  const ARGS: &'static CStr = cstr!(
    "clip:vnode;\
    window_size:int:opt;\
    topk:float:opt;\
    tvi_threshold:float:opt;\
    max_log_contrast:int:opt;\
    prop:data:opt;"
  );
  const RETURN_TYPE: &'static CStr = cstr!("clip:vnode;");
}

declare_plugin!(
  "sgt.cambi",
  "cambi",
  "Contrast Aware Multiscale Banding Index (CAMBI).",
  (1, 0),
  VAPOURSYNTH_API_VERSION,
  0,
  (CambiFilter, None)
);
