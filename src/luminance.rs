use kolor_64::details::transform::ST_2084_PQ_eotf_float;
use num_traits::clamp;
use vapoursynth4_rs::ffi::VSColorRange;

/// Contains the necessary information to normalize a luma value down to `[0, 1]`.
#[derive(Debug)]
pub struct LumaRange {
  pub foot: i32,
  pub head: i32,
}

impl LumaRange {
  pub const fn new(bit_depth: i32, pix_range: VSColorRange) -> Self {
    let (foot, head) = match pix_range {
      VSColorRange::VSC_RANGE_FULL => (0, (1 << bit_depth) - 1),
      VSColorRange::VSC_RANGE_LIMITED => (16 * (1 << (bit_depth - 8)), 235 * (1 << (bit_depth - 8))),
    };
    Self { foot, head }
  }
}

impl PartialEq<(i32, i32)> for LumaRange {
  fn eq(&self, other: &(i32, i32)) -> bool {
    self.foot == other.0 && self.head == other.1
  }
}

pub fn normalize_range(sample: i32, range: &LumaRange) -> f64 {
  let clipped_sample = clamp(sample, range.foot, range.head);
  f64::from(clipped_sample - range.foot) / f64::from(range.head - range.foot)
}

/// Electro-optical transfer function.
#[derive(Debug)]
pub enum Eotf {
  /// ITU-R BT.1886.
  Bt1886,

  /// Perceptual quantizer (SMPTE ST 2084).
  Pq,
}

const BT1886_GAMMA: f64 = 2.4;
const BT1886_LB: f64 = 0.01;
const BT1886_LW: f64 = 300.0;

fn bt_1866_eotf(v: f64) -> f64 {
  let a = (BT1886_LW.powf(1.0 / BT1886_GAMMA) - BT1886_LB.powf(1.0 / BT1886_GAMMA)).powf(BT1886_GAMMA);
  let b =
    BT1886_LB.powf(1.0 / BT1886_GAMMA) / (BT1886_LW.powf(1.0 / BT1886_GAMMA) - BT1886_LB.powf(1.0 / BT1886_GAMMA));
  a * f64::max(v + b, 0.0).powf(BT1886_GAMMA)
}

pub fn get_luminance(sample: i32, luma_range: &LumaRange, eotf: &Eotf) -> f64 {
  let normalized = normalize_range(sample, luma_range);
  match eotf {
    Eotf::Bt1886 => bt_1866_eotf(normalized),
    Eotf::Pq => ST_2084_PQ_eotf_float(normalized),
  }
}

#[cfg(test)]
mod tests {
  use approx::assert_relative_eq;

  use super::*;

  const EPISILON: f64 = 0.000000000001;

  #[test]
  fn test_luma_range() {
    assert_eq!(LumaRange::new(8, VSColorRange::VSC_RANGE_LIMITED), (16, 235));
    assert_eq!(LumaRange::new(10, VSColorRange::VSC_RANGE_LIMITED), (64, 940));
    assert_eq!(LumaRange::new(8, VSColorRange::VSC_RANGE_FULL), (0, 255));
    assert_eq!(LumaRange::new(10, VSColorRange::VSC_RANGE_FULL), (0, 1023));
  }

  #[test]
  fn test_bt1886_eof() {
    assert_relative_eq!(bt_1866_eotf(0.5), 58.716634039821685, epsilon = EPISILON);
    assert_relative_eq!(bt_1866_eotf(0.1), 1.5766526614315794, epsilon = EPISILON);
    assert_relative_eq!(bt_1866_eotf(0.9), 233.81950301956385, epsilon = EPISILON);
  }

  // Doesn't test our code, but this is here anyways to ensure that the output
  // matches what libvmaf has.
  #[test]
  fn test_pq_eof() {
    assert_relative_eq!(ST_2084_PQ_eotf_float(0.0), 0.0);
    assert_relative_eq!(ST_2084_PQ_eotf_float(0.1), 0.3245655914644875, epsilon = EPISILON);
    assert_relative_eq!(ST_2084_PQ_eotf_float(0.3), 10.038226310511204, epsilon = EPISILON);
    assert_relative_eq!(ST_2084_PQ_eotf_float(0.8), 1555.1783642892847, epsilon = EPISILON);
  }

  #[test]
  fn test_get_luminance() {
    assert_relative_eq!(
      get_luminance(400, &LumaRange::new(10, VSColorRange::VSC_RANGE_LIMITED), &Eotf::Bt1886),
      31.68933962217197,
      epsilon = EPISILON
    );
    assert_relative_eq!(
      get_luminance(400, &LumaRange::new(10, VSColorRange::VSC_RANGE_FULL), &Eotf::Bt1886),
      33.13300375755777,
      epsilon = EPISILON
    );
  }
}
