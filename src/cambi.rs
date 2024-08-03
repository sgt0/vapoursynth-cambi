use std::{
  cmp::{max, min},
  ops::{AddAssign, SubAssign},
};

use num_traits::{clamp, One};
use vapoursynth4_rs::frame::VideoFrame;

use crate::luminance::{get_luminance, Eotf, LumaRange};

const CONTRAST_WEIGHTS: [u16; 32] = [
  1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
];
const MASK_FILTER_SIZE: u16 = 7;
const NUM_SCALES: usize = 5;
const SCALE_WEIGHTS: [u16; NUM_SCALES] = [16, 8, 4, 2, 1];

#[derive(Debug, PartialEq)]
enum TviBisect {
  TooSmall,
  Correct,
  TooBig,
}

fn tvi_condition(sample: i32, diff: i32, tvi_threshold: f64, luma_range: &LumaRange, eotf: &Eotf) -> bool {
  let mean_luminance = get_luminance(sample, luma_range, eotf);
  let diff_luminance = get_luminance(sample + diff, luma_range, eotf);
  let delta_luminance = diff_luminance - mean_luminance;
  delta_luminance > tvi_threshold * mean_luminance
}

fn tvi_hard_threshold_condition(
  sample: i32,
  diff: i32,
  tvi_threshold: f64,
  luma_range: &LumaRange,
  eotf: &Eotf,
) -> TviBisect {
  if !tvi_condition(sample, diff, tvi_threshold, luma_range, eotf) {
    return TviBisect::TooBig;
  }

  if tvi_condition(sample + 1, diff, tvi_threshold, luma_range, eotf) {
    return TviBisect::TooSmall;
  }

  TviBisect::Correct
}

pub fn get_tvi_for_diff(diff: i32, tvi_threshold: f64, bit_depth: i32, luma_range: &LumaRange, eotf: &Eotf) -> i32 {
  let (mut foot, mut head) = (luma_range.foot, luma_range.head);
  head = head - diff - 1;

  let tvi_bisect = tvi_hard_threshold_condition(foot, diff, tvi_threshold, luma_range, eotf);
  match tvi_bisect {
    TviBisect::TooBig => {
      return 0;
    }
    TviBisect::Correct => {
      return foot;
    }
    TviBisect::TooSmall => { /* fallthrough */ }
  };

  let tvi_bisect = tvi_hard_threshold_condition(head, diff, tvi_threshold, luma_range, eotf);
  match tvi_bisect {
    TviBisect::TooSmall => {
      return (1 << bit_depth) - 1; // Maximum value.
    }
    TviBisect::Correct => {
      return head;
    }
    TviBisect::TooBig => { /* fallthrough */ }
  };

  // Bisect.
  loop {
    let mid = foot + (head - foot) / 2;
    let tvi_bisect = tvi_hard_threshold_condition(mid, diff, tvi_threshold, luma_range, eotf);
    match tvi_bisect {
      TviBisect::TooBig => {
        head = mid;
      }
      TviBisect::TooSmall => {
        foot = mid;
      }
      TviBisect::Correct => {
        return mid;
      }
    }
  }
}

pub const fn adjust_window_size(window_size: u16, input_width: i32, input_height: i32) -> u16 {
  (((window_size as i32 * (input_width + input_height)) / 375) >> 4) as u16
}

pub struct ContrastArrays {
  pub diffs_to_consider: Vec<u16>,
  pub diff_weights: Vec<i32>,
  pub all_diffs: Vec<i32>,
}

impl ContrastArrays {
  pub fn new(num_diffs: i32) -> Self {
    let mut diffs_to_consider = Vec::with_capacity(num_diffs as usize);
    let mut diff_weights = Vec::with_capacity(num_diffs as usize);
    for d in 0..num_diffs {
      diffs_to_consider.push(d as u16 + 1);
      diff_weights.push(i32::from(CONTRAST_WEIGHTS[d as usize]));
    }

    let mut all_diffs = Vec::with_capacity((2 * num_diffs + 1) as usize);
    for d in -num_diffs..=num_diffs {
      all_diffs.push(d);
    }

    Self {
      diffs_to_consider,
      diff_weights,
      all_diffs,
    }
  }
}

fn increment_range<T>(arr: &mut [T], left: i32, right: i32)
where
  T: AddAssign + One,
{
  for x in &mut arr[left as usize..right as usize] {
    *x += T::one();
  }
}

fn decrement_range<T>(arr: &mut [T], left: i32, right: i32)
where
  T: SubAssign + One,
{
  for x in &mut arr[left as usize..right as usize] {
    *x -= T::one();
  }
}

fn decimate(image: &mut VideoFrame, width: i32, height: i32) {
  let data: *mut u16 = image.plane_mut(0).cast();
  let stride = image.stride(0) >> 1;
  for i in 0..height as isize {
    for j in 0..width as isize {
      unsafe {
        *data.offset(i * stride + j) = *data.offset((i << 1) * stride + (j << 1));
      }
    }
  }
}

fn min3<T>(a: T, b: T, c: T) -> T
where
  T: Ord,
{
  min(min(a, b), c)
}

fn mode3<T>(a: T, b: T, c: T) -> T
where
  T: Ord,
{
  if a == b || a == c {
    return a;
  }
  if b == c {
    return b;
  }
  min3(a, b, c)
}

fn filter_mode(image: &mut VideoFrame, width: i32, height: i32, buffer: &mut [u16]) {
  let data: *mut u16 = image.plane_mut(0).cast();
  let stride = image.stride(0) >> 1;
  let mut curr_line = 0;
  for i in 0..height {
    unsafe {
      buffer[(curr_line * width) as usize] = *data.offset(i as isize * stride);
    }
    for j in 1..(width - 1) {
      unsafe {
        buffer[(curr_line * width + j) as usize] = mode3(
          *data.offset((i * stride as i32 + j - 1) as isize),
          *data.offset((i * stride as i32 + j) as isize),
          *data.offset((i * stride as i32 + j + 1) as isize),
        ) as u16;
      }
    }
    unsafe {
      buffer[(curr_line * width + width - 1) as usize] = *data.offset((i * stride as i32 + width - 1) as isize);
    }

    if i > 1 {
      for j in 0..width {
        unsafe {
          *data.offset(((i - 1) * stride as i32 + j) as isize) = mode3(
            buffer[(j) as usize],         // 0 * width + j
            buffer[(width + j) as usize], // 1 * width + j
            buffer[(2 * width + j) as usize],
          );
        }
      }
    }
    curr_line = if curr_line + 1 == 3 { 0 } else { curr_line + 1 };
  }
}

fn get_derivative_data_for_row(
  image_data: *const u16,
  derivative_buffer: &mut [u16],
  width: usize,
  height: usize,
  row: usize,
  stride: usize,
) {
  debug_assert!(width > 0);
  debug_assert!(height > 0);

  for (col, derivative) in derivative_buffer.iter_mut().enumerate() {
    unsafe {
      let horizontal_derivative =
        col == width - 1 || *image_data.add(row * stride + col) == *image_data.add(row * stride + col + 1);
      let vertical_derivative =
        row == height - 1 || *image_data.add(row * stride + col) == *image_data.add((row + 1) * stride + col);
      *derivative = u16::from(horizontal_derivative && vertical_derivative);
    }
  }
}

fn ceil_log2(num: u32) -> u16 {
  if num == 0 {
    return 0;
  }

  f64::from(num).log2().ceil() as u16
}

fn get_mask_index(input_width: u32, input_height: u32, filter_size: u16) -> u16 {
  let shifted_wh = (input_width >> 6) * (input_height >> 6);
  (((filter_size * filter_size) as i16 + 3 * (ceil_log2(shifted_wh) as i16 - 11) - 1) >> 1) as u16
}

fn get_spatial_mask_for_index(
  image: &VideoFrame,
  mask: &mut VideoFrame,
  derivative_buffer: &mut [u16],
  mask_index: u16,
  filter_size: u16,
  width: i32,
  height: i32,
) {
  let pad_size = filter_size >> 1;
  let image_data: *const u16 = image.plane(0).cast();
  let mask_data: *mut u16 = mask.plane_mut(0).cast();
  let stride = image.stride(0) as i32 >> 1;

  let dp_width = width + 2 * i32::from(pad_size) + 1;
  let dp_height = i32::from(2 * pad_size + 2);
  let mut dp = vec![0; (dp_width * dp_height) as usize];

  // Initial computation: fill dp except for the last row.
  for i in 0..pad_size {
    if i32::from(i) < height {
      get_derivative_data_for_row(
        image_data,
        derivative_buffer,
        width as usize,
        height as usize,
        i as usize,
        stride as usize,
      );
    }
    for j in 0..(width + i32::from(pad_size)) {
      let value = if i32::from(i) < height && j < width {
        derivative_buffer[j as usize]
      } else {
        0
      };
      let curr_row = i32::from(i + pad_size + 1);
      let curr_col = j + i32::from(pad_size) + 1;
      dp[(curr_row * dp_width + curr_col) as usize] = u32::from(value)
        + dp[((curr_row - 1) * dp_width + curr_col) as usize]
        + dp[(curr_row * dp_width + curr_col - 1) as usize]
        - dp[((curr_row - 1) * dp_width + curr_col - 1) as usize];
    }
  }

  // Start from the last row in the dp matrix.
  let mut prev_row = dp_height - 2;
  let mut curr_row = dp_height - 1;
  let mut curr_compute = i32::from(pad_size + 1);
  let mut bottom = (curr_compute + i32::from(pad_size)) % dp_height;
  let mut top = (curr_compute + dp_height - i32::from(pad_size) - 1) % dp_height;
  for i in pad_size..(height as u16 + pad_size) {
    if i32::from(i) < height {
      get_derivative_data_for_row(
        image_data,
        derivative_buffer,
        width as usize,
        height as usize,
        i as usize,
        stride as usize,
      );
    }

    // First compute the values of dp for curr_row.
    for j in 0..(width + i32::from(pad_size)) {
      let value = if i32::from(i) < height && j < width {
        derivative_buffer[j as usize]
      } else {
        0
      };

      let curr_col = j + i32::from(pad_size) + 1;
      dp[(curr_row * dp_width + curr_col) as usize] = u32::from(value)
        + dp[(prev_row * dp_width + curr_col) as usize]
        + dp[(curr_row * dp_width + curr_col - 1) as usize]
        - dp[(prev_row * dp_width + curr_col - 1) as usize];
    }
    prev_row = curr_row;
    curr_row = if curr_row + 1 == dp_height { 0 } else { curr_row + 1 };

    // Then use the values to compute the square sum for the `curr_compute` row.
    for j in 0..width {
      let curr_col = j + i32::from(pad_size) + 1;
      let right = curr_col + i32::from(pad_size);
      let left = curr_col - i32::from(pad_size) - 1;

      // May go negative.
      let result = dp[(bottom * dp_width + right) as usize] as i32
        - dp[(bottom * dp_width + left) as usize] as i32
        - dp[(top * dp_width + right) as usize] as i32
        + dp[(top * dp_width + left) as usize] as i32;
      unsafe {
        *mask_data.offset((i32::from(i - pad_size) * stride + j) as isize) = u16::from(result > i32::from(mask_index));
      }
    }
    curr_compute = if curr_compute + 1 == dp_height {
      0
    } else {
      curr_compute + 1
    };
    bottom = if bottom + 1 == dp_height { 0 } else { bottom + 1 };
    top = if top + 1 == dp_height { 0 } else { top + 1 };
  }
}

fn get_spatial_mask(image: &VideoFrame, mask: &mut VideoFrame, derivative_buffer: &mut [u16], width: i32, height: i32) {
  let mask_index = get_mask_index(width as u32, height as u32, MASK_FILTER_SIZE);
  get_spatial_mask_for_index(
    image,
    mask,
    derivative_buffer,
    mask_index,
    MASK_FILTER_SIZE,
    width,
    height,
  );
}

#[allow(clippy::cast_precision_loss)]
fn c_value_pixel(
  histograms: &[u16],
  value: u16,
  diff_weights: &[i32],
  diffs: &[i32],
  num_diffs: u16,
  tvi_thresholds: &[u16],
  histogram_col: i32,
  histogram_width: i32,
) -> f32 {
  let num_diffs = num_diffs as usize;

  let p_0 = histograms[(i32::from(value) * histogram_width + histogram_col) as usize];
  let mut c_value = 0.0;

  for d in 0..num_diffs {
    if value <= tvi_thresholds[d] {
      let p_1 = histograms[((i32::from(value) + diffs[num_diffs + d + 1]) * histogram_width + histogram_col) as usize];
      let p_2 = histograms[((i32::from(value) + diffs[num_diffs - d - 1]) * histogram_width + histogram_col) as usize];

      let val = if p_1 > p_2 {
        (diff_weights[d] * i32::from(p_0) * i32::from(p_1)) as f32 / f32::from(p_1 + p_0)
      } else {
        (diff_weights[d] * i32::from(p_0) * i32::from(p_2)) as f32 / f32::from(p_2 + p_0)
      };

      if val > c_value {
        c_value = val;
      }
    }
  }

  c_value
}

fn update_histogram_subtract_edge(
  histograms: &mut [u16],
  image: *const u16,
  mask: *const u16,
  i: i32,
  j: i32,
  width: i32,
  stride: isize,
  pad_size: u16,
  num_diffs: u16,
) {
  let count = ((i - i32::from(pad_size) - 1) * stride as i32 + j) as isize;
  let mask_val = unsafe { *mask.offset(count) };
  if mask_val > 0 {
    let val = unsafe { *image.offset(count) } + num_diffs;
    decrement_range(
      &mut histograms[(i32::from(val) * width) as usize..],
      max(j - i32::from(pad_size), 0),
      min(j + i32::from(pad_size) + 1, width),
    );
  }
}

fn update_histogram_subtract(
  histograms: &mut [u16],
  image: *const u16,
  mask: *const u16,
  i: i32,
  j: i32,
  width: i32,
  stride: isize,
  pad_size: u16,
  num_diffs: u16,
) {
  let pad_size = i32::from(pad_size);
  let count = ((i - pad_size - 1) * stride as i32 + j) as isize;
  let mask_val = unsafe { *mask.offset(count) };
  if mask_val > 0 {
    let val = unsafe { *image.offset(count) } + num_diffs;
    decrement_range(
      &mut histograms[(i32::from(val) * width) as usize..],
      j - pad_size,
      j + pad_size + 1,
    );
  }
}

fn update_histogram_add_edge(
  histograms: &mut [u16],
  image: *const u16,
  mask: *const u16,
  i: i32,
  j: i32,
  width: i32,
  stride: isize,
  pad_size: u16,
  num_diffs: u16,
) {
  let pad_size = i32::from(pad_size);
  let stride = stride as i32;
  let count = ((i + pad_size) * stride + j) as isize;
  let mask_val = unsafe { *mask.offset(count) };
  if mask_val > 0 {
    let val = unsafe { *image.offset(count) } + num_diffs;
    increment_range(
      &mut histograms[(i32::from(val) * width) as usize..],
      max(j - pad_size, 0),
      min(j + pad_size + 1, width),
    );
  }
}

fn update_histogram_add(
  histograms: &mut [u16],
  image: *const u16,
  mask: *const u16,
  i: i32,
  j: i32,
  width: i32,
  stride: isize,
  pad_size: u16,
  num_diffs: u16,
) {
  let pad_size = i32::from(pad_size);
  let stride = stride as i32;
  let count = ((i + pad_size) * stride + j) as isize;
  let mask_val = unsafe { *mask.offset(count) };
  if mask_val > 0 {
    let val = unsafe { *image.offset(count) } + num_diffs;
    increment_range(
      &mut histograms[(i32::from(val) * width) as usize..],
      max(j - pad_size, 0),
      min(j + pad_size + 1, width),
    );
  }
}

fn update_histogram_add_edge_first_pass(
  histograms: &mut [u16],
  image: *const u16,
  mask: *const u16,
  i: i32,
  j: i32,
  width: i32,
  stride: isize,
  pad_size: u16,
  num_diffs: u16,
) {
  let count = (i * stride as i32 + j) as isize;
  let mask_val = unsafe { *mask.offset(count) };
  if mask_val > 0 {
    let val = unsafe { *image.offset(count) } + num_diffs;
    increment_range(
      &mut histograms[(i32::from(val) * width) as usize..],
      max(j - i32::from(pad_size), 0),
      min(j + i32::from(pad_size) + 1, width),
    );
  }
}

fn update_histogram_add_first_pass(
  histograms: &mut [u16],
  image: *const u16,
  mask: *const u16,
  i: i32,
  j: i32,
  width: i32,
  stride: isize,
  pad_size: u16,
  num_diffs: u16,
) {
  let mask_val = unsafe { *mask.offset((i * stride as i32 + j) as isize) };
  if mask_val > 0 {
    let val = unsafe { *image.offset((i * stride as i32 + j) as isize) } + num_diffs;
    increment_range(
      &mut histograms[(i32::from(val) * width) as usize..],
      j - i32::from(pad_size),
      j + i32::from(pad_size) + 1,
    );
  }
}

fn calculate_c_values_row(
  c_values: &mut [f32],
  histograms: &[u16],
  image: *const u16,
  mask: *const u16,
  row: i32,
  width: i32,
  stride: isize,
  num_diffs: u16,
  tvi_for_diff: &[u16],
  diff_weights: &[i32],
  all_diffs: &[i32],
) {
  for col in 0..width {
    unsafe {
      if *mask.add((row * stride as i32 + col) as usize) > 0 {
        c_values[(row * width + col) as usize] = c_value_pixel(
          histograms,
          num_diffs + *image.add((row * stride as i32 + col) as usize),
          diff_weights,
          all_diffs,
          num_diffs,
          tvi_for_diff,
          col,
          width,
        );
      }
    }
  }
}

#[rustfmt::skip]
fn calculate_c_values(
  pic: &VideoFrame,
  mask: &VideoFrame,
  window_size: u16,
  num_diffs: u16,
  tvi_for_diff: &[u16],
  diff_weights: &[i32],
  all_diffs: &[i32],
  width: i32,
  height: i32,
) -> Vec<f32> {
  let pad_size = i32::from(window_size >> 1);
  let num_bins = 1024 + (all_diffs[(2 * num_diffs) as usize] - all_diffs[0]);

  let image: *const u16 = pic.plane(0).cast();
  let mask: *const u16 = mask.plane(0).cast();
  let stride = pic.stride(0) >> 1;

  let mut c_values = vec![0.0; (width * height) as usize];

  let mut histograms = vec![0; (width * num_bins) as usize];

  // First pass: first `pad_size` rows.
  for i in 0..pad_size {
    for j in 0..pad_size {
      update_histogram_add_edge_first_pass(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
    }
    for j in pad_size..(width - pad_size - 1) {
      update_histogram_add_first_pass(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
    }
    for j in max(width - pad_size - 1, pad_size)..width {
      update_histogram_add_edge_first_pass(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
    }
  }

  // Iterate over all rows, unrolled into 3 loops to avoid conditions.
  for i in 0..=pad_size {
    if i + pad_size < height {
        for j in 0..pad_size {
            update_histogram_add_edge(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
        }
        for j in pad_size..width - pad_size - 1 {
            update_histogram_add(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
        }
        for j in max(width - pad_size - 1, pad_size)..width {
            update_histogram_add_edge(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
        }
    }
    calculate_c_values_row(&mut c_values, &histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff, diff_weights, all_diffs);
  }
  for i in (pad_size + 1)..(height - pad_size) {
    for j in 0..pad_size {
          update_histogram_subtract_edge(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
          update_histogram_add_edge(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
      }
      for j in pad_size..width - pad_size - 1 {
          update_histogram_subtract(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
          update_histogram_add(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
      }
      for j in max(width - pad_size - 1, pad_size)..width {
          update_histogram_subtract_edge(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
          update_histogram_add_edge(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
      }
      calculate_c_values_row(&mut c_values, &histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff, diff_weights, all_diffs);
  }
  for i in height - pad_size..height {
    if i - pad_size > 0 {
      for j in 0..pad_size {
        update_histogram_subtract_edge(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
      }
      for j in pad_size..width - pad_size - 1 {
        update_histogram_subtract(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
      }
      for j in max(width - pad_size - 1, pad_size)..width {
        update_histogram_subtract_edge(&mut histograms, image, mask, i, j, width, stride, pad_size as u16, num_diffs);
      }
    }
    calculate_c_values_row(&mut c_values, &histograms, image, mask, i, width, stride, num_diffs, tvi_for_diff, diff_weights, all_diffs);
  }

  c_values
}

#[allow(clippy::cast_precision_loss)]
fn average_topk_elements(arr: &[f32], topk_elements: usize) -> f64 {
  f64::from(arr[0..topk_elements].iter().sum::<f32>()) / topk_elements as f64
}

#[allow(clippy::cast_precision_loss)]
fn spatial_pooling(c_values: &mut [f32], topk: f32, width: i32, height: i32) -> f64 {
  let num_elements = height * width;
  let topk_num_elements = clamp((topk * num_elements as f32) as i32, 1, num_elements);

  // Only sort if the the top `k` elements don't take up the whole slice. This
  // case typically doesn't happen as it would require `topk == 1.0`.
  if num_elements != topk_num_elements {
    c_values.select_nth_unstable_by(topk_num_elements as usize, |a, b| f32::total_cmp(b, a));
  }

  average_topk_elements(c_values, topk_num_elements as usize)
}

const fn get_pixels_in_window(window_length: u16) -> u16 {
  let odd_length = 2 * (window_length >> 1) + 1;
  odd_length * odd_length
}

/// Inner product weighting scores for each scale.
fn weight_scores_per_scale(scores_per_scale: &[f64], normalization: u16) -> f64 {
  let mut score: f64 = 0.0;
  for scale in 0..NUM_SCALES {
    score += scores_per_scale[scale] * f64::from(SCALE_WEIGHTS[scale]);
  }
  score / f64::from(normalization)
}

/// CAMBI parameters that are constant across frames.
pub struct CambiParams {
  pub contrast_arrays: ContrastArrays,
  pub num_diffs: i32,

  /// Ratio of pixels for the spatial pooling computation.
  pub topk: f32,

  /// Window size to compute CAMBI. Note that this instance of it is assumed to
  /// have been adjusted for the input resolution.
  pub window_size: u16,

  /// Input width.
  pub width: i32,

  /// Input height.
  pub height: i32,
}

pub fn cambi_score(image: &mut VideoFrame, mask: &mut VideoFrame, tvi_for_diff: &[u16], params: &CambiParams) -> f64 {
  let width = params.width;
  let height = params.height;

  let contrast_arrays = &params.contrast_arrays;
  let num_diffs = params.num_diffs;
  let topk = params.topk;
  let window_size = params.window_size;

  let mut derivative_buffer = vec![0; width as usize];
  let mut filter_mode_buffer = vec![0; (3 * width) as usize];

  let mut scores_per_scale = vec![0.0; NUM_SCALES];

  let mut scaled_width = width;
  let mut scaled_height = height;

  get_spatial_mask(image, mask, &mut derivative_buffer, width, height);
  for (scale, scores) in scores_per_scale.iter_mut().enumerate() {
    if scale > 0 {
      scaled_width = (scaled_width + 1) >> 1;
      scaled_height = (scaled_height + 1) >> 1;
      decimate(image, scaled_width, scaled_height);
      decimate(mask, scaled_width, scaled_height);
    }

    filter_mode(image, scaled_width, scaled_height, &mut filter_mode_buffer);

    let mut c_values = calculate_c_values(
      image,
      mask,
      window_size,
      num_diffs as u16,
      tvi_for_diff,
      &contrast_arrays.diff_weights,
      &contrast_arrays.all_diffs,
      scaled_width,
      scaled_height,
    );

    *scores = spatial_pooling(&mut c_values, topk, scaled_width, scaled_height);
  }

  weight_scores_per_scale(&scores_per_scale, get_pixels_in_window(window_size))
}

#[cfg(test)]
mod tests {
  use approx::assert_relative_eq;
  use vapoursynth4_rs::ffi::VSColorRange;

  use super::*;

  const EPSILON: f32 = 0.000000000001;

  #[test]
  fn test_get_mask_index() {
    assert_eq!(get_mask_index(3840, 2160, 7), 24);
    assert_eq!(get_mask_index(2560, 1440, 7), 22);
    assert_eq!(get_mask_index(1920, 1080, 7), 21);
  }

  #[test]
  fn test_c_value_pixel() {
    let histogram = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let value = 2;
    let diffs = [-2, -1, 0, 1, 2];
    let tvi_thresholds = [2, 3];
    let mut diff_weights = [1, 2];
    let num_diffs = 2;

    let c_value = c_value_pixel(
      &histogram,
      value,
      &diff_weights,
      &diffs,
      num_diffs,
      &tvi_thresholds,
      0,
      1,
    );
    assert_relative_eq!(c_value, 2.6666667, epsilon = EPSILON);

    diff_weights[0] = 4;
    diff_weights[1] = 5;
    let c_value = c_value_pixel(
      &histogram,
      value,
      &diff_weights,
      &diffs,
      num_diffs,
      &tvi_thresholds,
      0,
      1,
    );
    assert_relative_eq!(c_value, 6.6666667, epsilon = EPSILON);

    let value = 4;
    let c_value = c_value_pixel(
      &histogram,
      value,
      &diff_weights,
      &diffs,
      num_diffs,
      &tvi_thresholds,
      0,
      1,
    );
    assert_relative_eq!(c_value, 0.0);
  }

  #[test]
  fn test_spatial_pooling() {
    let mut arr = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 7.0, 8.0, 9.0, 6.0, 11.0];
    assert_relative_eq!(spatial_pooling(&mut arr, 0.0, 4, 3), 11.0, epsilon = f64::from(EPSILON));
    assert_relative_eq!(spatial_pooling(&mut arr, 0.1, 4, 3), 11.0, epsilon = f64::from(EPSILON));
    assert_relative_eq!(spatial_pooling(&mut arr, 0.2, 4, 3), 10.5, epsilon = f64::from(EPSILON));
    assert_relative_eq!(spatial_pooling(&mut arr, 1.0, 4, 3), 5.5, epsilon = f64::from(EPSILON));
  }

  #[test]
  fn test_adjust_window_size() {
    assert_eq!(adjust_window_size(63, 3840, 2160), 63);
    assert_eq!(adjust_window_size(63, 2560, 1440), 42);
    assert_eq!(adjust_window_size(63, 1920, 1080), 31);
  }

  #[test]
  fn test_weight_scores_per_scale() {
    let scores_per_scale = [10000.0, 1000.0, 100.0, 10.0, 1.0];
    assert_relative_eq!(weight_scores_per_scale(&scores_per_scale, 10), 16842.1);
  }

  #[test]
  fn test_tvi_hard_threshold_condition() {
    let limited = LumaRange::new(10, VSColorRange::VSC_RANGE_LIMITED);
    assert_eq!(
      tvi_hard_threshold_condition(177, 1, 0.019, &limited, &Eotf::Bt1886),
      TviBisect::TooSmall
    );
    assert_eq!(
      tvi_hard_threshold_condition(178, 1, 0.019, &limited, &Eotf::Bt1886),
      TviBisect::Correct
    );
    assert_eq!(
      tvi_hard_threshold_condition(179, 1, 0.019, &limited, &Eotf::Bt1886),
      TviBisect::TooBig
    );
    assert_eq!(
      tvi_hard_threshold_condition(305, 2, 0.019, &limited, &Eotf::Bt1886),
      TviBisect::Correct
    );
  }

  #[test]
  fn test_tvi_condition() {
    assert!(tvi_condition(
      177,
      1,
      0.019,
      &LumaRange::new(10, VSColorRange::VSC_RANGE_LIMITED),
      &Eotf::Bt1886
    ));

    assert!(tvi_condition(
      178,
      1,
      0.019,
      &LumaRange::new(10, VSColorRange::VSC_RANGE_LIMITED),
      &Eotf::Bt1886
    ));

    assert!(!tvi_condition(
      179,
      1,
      0.019,
      &LumaRange::new(10, VSColorRange::VSC_RANGE_LIMITED),
      &Eotf::Bt1886
    ));

    assert!(tvi_condition(
      935,
      4,
      0.01,
      &LumaRange::new(10, VSColorRange::VSC_RANGE_LIMITED),
      &Eotf::Bt1886
    ));

    assert!(tvi_condition(
      936,
      4,
      0.01,
      &LumaRange::new(10, VSColorRange::VSC_RANGE_LIMITED),
      &Eotf::Bt1886
    ));
  }
}
