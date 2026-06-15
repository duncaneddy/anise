/*
 * ANISE Toolkit
 * Copyright (C) 2021-onward Christopher Rabotin <christopher.rabotin@gmail.com> et al. (cf. AUTHORS.md)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * Documentation: https://nyxspace.com/
 */

use std::collections::HashMap;
use std::sync::RwLock;

use hifitime::{Duration, Epoch};

use crate::NaifId;
use crate::naif::daf::DafDataType;

/// Memoized location of the most recent segment match for a NAIF ID.
/// Indices are only hints: every use re-fetches the live summary and validates
/// the ID and epoch bounds before trusting it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpkSegmentMemo {
    pub spk_no: usize,
    pub daf_idx: Option<usize>,
    pub idx_in_spk: usize,
    /// Byte range of the summaries portion of the segment's summary record (i.e. past the
    /// SummaryRecord header), resolved at store time so a hit never re-parses the file record.
    pub summaries_byte_range: (usize, usize),
    /// Bounds of the memoized segment in ET seconds, copied from the summary at store time.
    /// Hit validation compares tightly (no ±100 ns pad): boundary epochs fall back to the
    /// full search, which applies the exact padded semantics.
    pub start_et_s: f64,
    pub end_et_s: f64,
    pub decoded: Option<DecodedSplineHeader>,
}

/// The parsed header of an interpolation set, memoized so repeated queries skip
/// `nth_data`'s file-record parse and header decode. Holds values only — no
/// references into the kernel bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DecodedSplineHeader {
    pub dtype: DafDataType,
    pub init_epoch: Epoch,
    pub interval_length: Duration,
    pub rsize: usize,
    pub num_records: usize,
    /// Byte range of the segment's f64 data within the kernel, resolved at store time.
    pub dbl_byte_range: (usize, usize),
}

/// Internal ephemeris query cache. A pure optimization: a poisoned lock or any
/// validation failure degrades to a cache miss, never to an error or a wrong result.
#[derive(Debug, Default)]
pub(crate) struct EphemerisCache {
    state: RwLock<CacheState>,
}

#[derive(Debug, Default)]
struct CacheState {
    root: Option<NaifId>,
    spk_segments: HashMap<NaifId, SpkSegmentMemo>,
}

/// Cloning yields a fresh, cold cache: clones are typically sent to other threads,
/// and sharing the lock would couple their performance.
impl Clone for EphemerisCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl EphemerisCache {
    pub(crate) fn spk_segment(&self, id: NaifId) -> Option<SpkSegmentMemo> {
        self.state.read().ok()?.spk_segments.get(&id).copied()
    }

    pub(crate) fn store_spk_segment(&self, id: NaifId, memo: SpkSegmentMemo) {
        if let Ok(mut state) = self.state.write() {
            state.spk_segments.insert(id, memo);
        }
    }

    /// Attaches a decoded header to the memo for `id`, but only if that memo still
    /// points at the exact segment the header was decoded from.
    pub(crate) fn store_decoded(
        &self,
        id: NaifId,
        spk_no: usize,
        daf_idx: Option<usize>,
        idx_in_spk: usize,
        header: DecodedSplineHeader,
    ) {
        if let Ok(mut state) = self.state.write()
            && let Some(memo) = state.spk_segments.get_mut(&id)
            && memo.spk_no == spk_no
            && memo.daf_idx == daf_idx
            && memo.idx_in_spk == idx_in_spk
        {
            memo.decoded = Some(header);
        }
    }

    pub(crate) fn root(&self) -> Option<NaifId> {
        self.state.read().ok()?.root
    }

    pub(crate) fn store_root(&self, root: NaifId) {
        if let Ok(mut state) = self.state.write() {
            state.root = Some(root);
        }
    }

    /// Drops all memoized data. Must be called whenever the set of loaded SPKs changes.
    pub(crate) fn invalidate(&self) {
        if let Ok(mut state) = self.state.write() {
            *state = CacheState::default();
        }
    }
}

#[cfg(test)]
mod ut_cache {
    use super::*;
    use crate::naif::daf::DafDataType;
    use hifitime::{Epoch, Unit};

    fn header() -> DecodedSplineHeader {
        DecodedSplineHeader {
            dtype: DafDataType::Type2ChebyshevTriplet,
            init_epoch: Epoch::from_et_seconds(0.0),
            interval_length: Unit::Day * 16,
            rsize: 41,
            num_records: 100,
            dbl_byte_range: (1024, 2048),
        }
    }

    fn memo(spk_no: usize, idx_in_spk: usize) -> SpkSegmentMemo {
        SpkSegmentMemo {
            spk_no,
            daf_idx: None,
            idx_in_spk,
            summaries_byte_range: (1024 + 24, 2048),
            start_et_s: -1.0e9,
            end_et_s: 1.0e9,
            decoded: None,
        }
    }

    #[test]
    fn store_and_lookup_segment() {
        let cache = EphemerisCache::default();
        assert!(cache.spk_segment(301).is_none());

        cache.store_spk_segment(301, memo(0, 3));
        let memo = cache.spk_segment(301).unwrap();
        assert_eq!(memo.spk_no, 0);
        assert_eq!(memo.idx_in_spk, 3);
        assert!(memo.decoded.is_none());
        assert!(cache.spk_segment(399).is_none());
    }

    #[test]
    fn store_decoded_requires_matching_location() {
        let cache = EphemerisCache::default();
        cache.store_spk_segment(301, memo(0, 3));

        // Mismatched location: must be ignored.
        cache.store_decoded(301, 1, None, 3, header());
        assert!(cache.spk_segment(301).unwrap().decoded.is_none());
        cache.store_decoded(301, 0, None, 4, header());
        assert!(cache.spk_segment(301).unwrap().decoded.is_none());
        // No memo at all for this ID: must be ignored.
        cache.store_decoded(399, 0, None, 3, header());
        assert!(cache.spk_segment(399).is_none());

        // Matching location: stored.
        cache.store_decoded(301, 0, None, 3, header());
        assert_eq!(cache.spk_segment(301).unwrap().decoded.unwrap().rsize, 41);
    }

    #[test]
    fn root_memo_and_invalidate() {
        let cache = EphemerisCache::default();
        assert!(cache.root().is_none());
        cache.store_root(0);
        assert_eq!(cache.root(), Some(0));

        cache.store_spk_segment(301, memo(0, 0));
        cache.invalidate();
        assert!(cache.root().is_none());
        assert!(cache.spk_segment(301).is_none());
    }

    #[test]
    fn clone_is_cold() {
        let cache = EphemerisCache::default();
        cache.store_root(0);
        let cloned = cache.clone();
        assert!(cloned.root().is_none());
        // Original is untouched.
        assert_eq!(cache.root(), Some(0));
    }
}
