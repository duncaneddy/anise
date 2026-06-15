use anise::{
    constants::frames::{EARTH_J2000, MOON_J2000},
    file2heap,
    prelude::*,
};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

/// Mirrors a numerical propagation workload: 1,440 sequential queries at 60 s steps.
fn benchmark_propagation_like(ctx: &Almanac, time_it: TimeSeries) {
    for epoch in time_it {
        black_box(
            ctx.translate_geometric(MOON_J2000, EARTH_J2000, epoch)
                .unwrap(),
        );
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let start_epoch = Epoch::from_gregorian_at_noon(2024, 1, 1, TimeScale::ET);
    let time_step = 60.0.seconds();
    let end_epoch = start_epoch + 1440.0 * time_step;
    let time_it = TimeSeries::exclusive(start_epoch, end_epoch, time_step);

    let path = "../data/de440s.bsp";
    let buf = file2heap!(path).unwrap();
    let spk = SPK::parse(buf).unwrap();
    let ctx = Almanac::from_spk(spk);

    c.bench_function("ANISE Moon->Earth propagation-like day", |b| {
        b.iter(|| benchmark_propagation_like(&ctx, time_it.clone()))
    });
}

criterion_group!(spk_cache, criterion_benchmark);
criterion_main!(spk_cache);
