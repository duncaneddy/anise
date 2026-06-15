/*
 * ANISE Toolkit
 * Copyright (C) 2021-onward Christopher Rabotin <christopher.rabotin@gmail.com> et al. (cf. AUTHORS.md)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * Documentation: https://nyxspace.com/
 */

use anise::constants::frames::{EARTH_J2000, MOON_J2000};
use anise::file2heap;
use anise::naif::daf::{NAIFDataSet, datatypes::Type2ChebyshevSet};
use anise::prelude::*;

const MOON: i32 = 301;

/// After a hit is memoized, loading a new kernel must restore SPICE precedence:
/// the most recently loaded kernel wins.
#[test]
fn cache_invalidation_respects_precedence() {
    let almanac = Almanac::new("../data/de440s.bsp").unwrap();
    let epoch = Epoch::from_gregorian_at_noon(2024, 1, 1, TimeScale::ET);

    // Warm the memo for the Moon: with one kernel loaded, spk_no must be 0.
    let (_, spk_no, _, _) = almanac.spk_summary_at_epoch(MOON, epoch).unwrap();
    assert_eq!(spk_no, 0);
    // Query again to ensure we are answering from the memo.
    let (_, spk_no, _, _) = almanac.spk_summary_at_epoch(MOON, epoch).unwrap();
    assert_eq!(spk_no, 0);

    // Load the same kernel under a new alias: it now takes precedence (spk_no 1).
    let bytes = file2heap!("../data/de440s.bsp").unwrap();
    let spk = SPK::parse(bytes).unwrap();
    let almanac = almanac.with_spk_as(spk, Some("copy".to_string()));
    let (_, spk_no, _, _) = almanac.spk_summary_at_epoch(MOON, epoch).unwrap();
    assert_eq!(
        spk_no, 1,
        "a stale memo must not override the precedence of a newly loaded kernel"
    );
}

/// When two kernels both cover an epoch for the same ID, the last-loaded kernel must
/// win on every query, even after a previous query could have memoized the segment of
/// the lower-precedence kernel (whose coverage also includes that epoch).
#[test]
fn cache_shadowing_last_loaded_kernel_wins() {
    // Kernel A: full de440s (the Moon covers ~1849..2150).
    let spk_a = SPK::load("../data/de440s.bsp").unwrap();

    // Kernel B: de440s with the Moon segment truncated to [2030-01-01, 2040-01-01].
    let new_start = Epoch::from_gregorian_at_midnight(2030, 1, 1, TimeScale::ET);
    let new_end = Epoch::from_gregorian_at_midnight(2040, 1, 1, TimeScale::ET);
    let (summary, _, idx) = spk_a.summary_from_id(MOON).unwrap();
    let summary = *summary;
    let segment = spk_a.nth_data::<Type2ChebyshevSet>(None, idx).unwrap();
    let truncated = segment
        .truncate(&summary, Some(new_start), Some(new_end))
        .unwrap();
    let mut spk_b = spk_a.clone();
    spk_b
        .set_nth_data(idx, truncated, new_start, new_end)
        .unwrap();
    // Persist and reload so the chronological index is rebuilt from the new bytes.
    let path = "../target/cache-shadowing-truncated-de440s.bsp";
    spk_b.persist(path).unwrap();
    let spk_b = SPK::load(path).unwrap();

    // Load A then B: B is the most recently loaded, so it has precedence (spk_no 1).
    let almanac = Almanac::from_spk(spk_a).with_spk_as(spk_b, Some("truncated".to_string()));

    let only_a = Epoch::from_gregorian_at_noon(2024, 1, 1, TimeScale::ET);
    let both = Epoch::from_gregorian_at_noon(2035, 1, 1, TimeScale::ET);

    // Only kernel A covers 2024.
    let (_, spk_no, _, _) = almanac.spk_summary_at_epoch(MOON, only_a).unwrap();
    assert_eq!(spk_no, 0, "only kernel A covers 2024");
    // Both kernels cover 2035: the last-loaded kernel (B) must win, even though A's
    // segment found by the previous query also covers 2035.
    let (_, spk_no, _, _) = almanac.spk_summary_at_epoch(MOON, both).unwrap();
    assert_eq!(
        spk_no, 1,
        "last-loaded kernel must win when both cover the epoch"
    );
    // Alternate again to ensure warm queries stay correct in both directions.
    let (_, spk_no, _, _) = almanac.spk_summary_at_epoch(MOON, only_a).unwrap();
    assert_eq!(spk_no, 0, "kernel A must still answer the 2024 query");
    let (_, spk_no, _, _) = almanac.spk_summary_at_epoch(MOON, both).unwrap();
    assert_eq!(spk_no, 1, "kernel B must still answer the 2035 query");
}

/// A memoized segment must never answer for an epoch outside its bounds.
#[test]
fn cache_epoch_validation_falls_back() {
    let almanac = Almanac::new("../data/de440s.bsp").unwrap();
    let inside = Epoch::from_gregorian_at_noon(2024, 1, 1, TimeScale::ET);
    // de440s ends in 2150.
    let outside = Epoch::from_gregorian_at_noon(2700, 1, 1, TimeScale::ET);

    assert!(almanac.spk_summary_at_epoch(MOON, inside).is_ok());
    assert!(
        almanac.spk_summary_at_epoch(MOON, outside).is_err(),
        "memo must not answer outside its epoch bounds"
    );
    // And the memo still answers the valid epoch afterward.
    assert!(almanac.spk_summary_at_epoch(MOON, inside).is_ok());
}

/// Querying forward and reverse over the same epochs (different miss/hit patterns)
/// must produce identical states, and must match a cold context spot-check.
#[test]
fn cache_transparency_forward_vs_reverse() {
    let fwd_ctx = Almanac::new("../data/de440s.bsp").unwrap();
    let rev_ctx = Almanac::new("../data/de440s.bsp").unwrap();

    let start = Epoch::from_gregorian_at_noon(2024, 1, 1, TimeScale::ET);
    let epochs: Vec<Epoch> = (0..1_440)
        .map(|ii| start + (ii as f64 * 60.0).seconds())
        .collect();

    let fwd: Vec<_> = epochs
        .iter()
        .map(|e| {
            fwd_ctx
                .translate_geometric(MOON_J2000, EARTH_J2000, *e)
                .unwrap()
        })
        .collect();
    let mut rev: Vec<_> = epochs
        .iter()
        .rev()
        .map(|e| {
            rev_ctx
                .translate_geometric(MOON_J2000, EARTH_J2000, *e)
                .unwrap()
        })
        .collect();
    rev.reverse();

    for (warm_a, warm_b) in fwd.iter().zip(&rev) {
        assert_eq!(warm_a.radius_km, warm_b.radius_km);
        assert_eq!(warm_a.velocity_km_s, warm_b.velocity_km_s);
    }

    // Spot-check three epochs against fully cold contexts (first query = uncached path).
    for idx in [0_usize, 719, 1_439] {
        let cold_ctx = Almanac::new("../data/de440s.bsp").unwrap();
        let cold = cold_ctx
            .translate_geometric(MOON_J2000, EARTH_J2000, epochs[idx])
            .unwrap();
        assert_eq!(fwd[idx].radius_km, cold.radius_km);
        assert_eq!(fwd[idx].velocity_km_s, cold.velocity_km_s);
    }
}

/// Concurrent queries against one shared Almanac must match serial results exactly.
#[test]
fn cache_concurrent_queries_match_serial() {
    use rayon::prelude::*;

    let start = Epoch::from_gregorian_at_noon(2024, 1, 1, TimeScale::ET);
    let epochs: Vec<Epoch> = (0..1_440)
        .map(|ii| start + (ii as f64 * 60.0).seconds())
        .collect();

    let serial_ctx = Almanac::new("../data/de440s.bsp").unwrap();
    let serial: Vec<_> = epochs
        .iter()
        .map(|e| {
            serial_ctx
                .translate_geometric(MOON_J2000, EARTH_J2000, *e)
                .unwrap()
        })
        .collect();

    let shared_ctx = Almanac::new("../data/de440s.bsp").unwrap();
    let parallel: Vec<_> = epochs
        .par_iter()
        .map(|e| {
            shared_ctx
                .translate_geometric(MOON_J2000, EARTH_J2000, *e)
                .unwrap()
        })
        .collect();

    for (s, p) in serial.iter().zip(&parallel) {
        assert_eq!(s.radius_km, p.radius_km);
        assert_eq!(s.velocity_km_s, p.velocity_km_s);
    }
}
