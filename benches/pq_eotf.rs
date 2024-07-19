use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kolor_64::details::transform::ST_2084_PQ_eotf_float;

const PQ_M_1: f64 = 0.1593017578125;
const PQ_M_2: f64 = 78.84375;
const PQ_C_1: f64 = 0.8359375;
const PQ_C_2: f64 = 18.8515625;
const PQ_C_3: f64 = 18.6875;

fn pq_eotf(v: f64) -> f64 {
  let num = v.powf(1.0 / PQ_M_2) - PQ_C_1;
  let num_clipped = f64::max(num, 0.0);
  // PQ_C_2 - PQ_C_3 * v.powf(1.0 / PQ_M_2);
  let den = PQ_C_3.mul_add(-v.powf(1.0 / PQ_M_2), PQ_C_2);
  10000.0 * (num_clipped / den).powf(1.0 / PQ_M_1)
}

fn bench_pq_eotf(c: &mut Criterion) {
  let mut group = c.benchmark_group("pq_eotf");
  for i in [0.0, 0.1, 0.3, 0.8].iter() {
    group.bench_with_input(BenchmarkId::new("libvmaf", i), i, |b, i| b.iter(|| pq_eotf(*i)));
    group.bench_with_input(BenchmarkId::new("kolor-64", i), i, |b, i| {
      b.iter(|| ST_2084_PQ_eotf_float(*i))
    });
  }
  group.finish();
}

criterion_group!(benches, bench_pq_eotf);
criterion_main!(benches);
