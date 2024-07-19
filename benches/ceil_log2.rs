use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num_traits::{ToPrimitive, Zero};

const fn ceil_log2_vmaf(num: u32) -> u16 {
  if num == 0 {
    return 0;
  }

  let mut tmp = num - 1;
  let mut shift = 0;
  while tmp > 0 {
    tmp >>= 1;
    shift += 1;
  }
  shift
}

fn ceil_log2_std(num: u32) -> u16 {
  if num == 0 {
    return 0;
  }

  f64::from(num).log2().ceil() as u16
}

fn ceil_log2_std_generic<T: ToPrimitive + Zero>(num: T) -> i16 {
  if num.is_zero() {
    return 0;
  }

  num.to_f64().unwrap().log2().ceil() as i16
}

fn bench_ceil_log2(c: &mut Criterion) {
  let mut group = c.benchmark_group("ceil_log2");
  for i in [480, 880, 1980].iter() {
    group.bench_with_input(BenchmarkId::new("libvmaf", i), i, |b, i| b.iter(|| ceil_log2_vmaf(*i)));
    group.bench_with_input(BenchmarkId::new("std", i), i, |b, i| b.iter(|| ceil_log2_std(*i)));
    group.bench_with_input(BenchmarkId::new("std_generic", i), i, |b, i| {
      b.iter(|| ceil_log2_std_generic(*i))
    });
  }
  group.finish();
}

criterion_group!(benches, bench_ceil_log2);
criterion_main!(benches);
