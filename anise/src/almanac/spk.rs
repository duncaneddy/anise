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

use hifitime::Epoch;

#[cfg(feature = "python")]
use pyo3::prelude::*;
use snafu::ensure;

use crate::ephemerides::NoEphemerisLoadedSnafu;
use crate::naif::SPK;
use crate::naif::daf::DAFError;
use crate::naif::daf::NAIFSummaryRecord;
use crate::naif::spk::summary::SPKSummaryRecord;
use crate::{NaifId, ephemerides::EphemerisError};
use log::{error, warn};

use super::Almanac;
use super::cache::SpkSegmentMemo;

impl Almanac {
    pub fn from_spk(spk: SPK) -> Self {
        let me = Self::default();
        me.with_spk(spk)
    }

    /// Loads a new SPK file into a new context, using the system time as the alias. If the time is not availble, then 0 TAI is used.
    /// This new context is needed to satisfy the unloading of files. In fact, to unload a file, simply let the newly loaded context drop out of scope and Rust will clean it up.
    pub fn with_spk(self, spk: SPK) -> Self {
        self.with_spk_as(spk, None)
    }

    /// Loads a new SPK file into a new context, naming it with the provided alias, or the current system time if no alias is provided.
    /// To unload a file, call spk_unload.
    pub fn with_spk_as(mut self, spk: SPK, alias: Option<String>) -> Self {
        // For lifetime reasons, we format the message using a ref first.
        // This message is only displayed if there was something with that name before.
        let alias = alias.unwrap_or(Epoch::now().unwrap_or_default().to_string());
        let msg = format!("unloading SPK `{alias}`");
        if self.spk_data.insert(alias, spk).is_some() {
            warn!("{msg}");
        }
        self.cache.invalidate();
        self
    }

    /// Unloads the SPK with the provided alias.
    /// **WARNING:** This causes the order of the loaded files to be perturbed, which may be an issue if several SPKs with the same IDs are loaded.
    pub fn spk_unload(&mut self, alias: &str) -> Result<(), EphemerisError> {
        if self.spk_data.swap_remove(alias).is_none() {
            Err(EphemerisError::AliasNotFound {
                alias: alias.to_string(),
                action: "unload ephemeris",
            })
        } else {
            self.cache.invalidate();
            Ok(())
        }
    }
}

impl Almanac {
    pub fn num_loaded_spk(&self) -> usize {
        self.spk_data.len()
    }

    /// Returns the summary given the name of the summary record if that summary has data defined at the requested epoch and the SPK where this name was found to be valid at that epoch.
    pub fn spk_summary_from_name_at_epoch(
        &self,
        name: &str,
        epoch: Epoch,
    ) -> Result<(&SPKSummaryRecord, usize, Option<usize>, usize), EphemerisError> {
        for (spk_no, spk) in self.spk_data.values().rev().enumerate() {
            if let Ok((summary, daf_idx, idx_in_spk)) = spk.summary_from_name_at_epoch(name, epoch)
            {
                // NOTE: We're iterating backward, so the correct SPK number is "total loaded" minus "current iteration".
                return Ok((
                    summary,
                    self.num_loaded_spk() - spk_no - 1,
                    daf_idx,
                    idx_in_spk,
                ));
            }
        }

        // If we're reached this point, there is no relevant summary at this epoch.
        error!("Almanac: No summary {name} valid at epoch {epoch}");
        Err(EphemerisError::SPK {
            action: "searching for SPK summary",
            source: DAFError::SummaryNameAtEpochError {
                kind: "SPK",
                name: name.to_string(),
                epoch,
            },
        })
    }

    /// Returns the summary given the ID of the summary record if that summary has data defined at the requested epoch
    pub fn spk_summary_at_epoch(
        &self,
        id: i32,
        epoch: Epoch,
    ) -> Result<(&SPKSummaryRecord, usize, Option<usize>, usize), EphemerisError> {
        // Fast path: validate the memoized location from the previous query for this ID.
        // The memo holds byte ranges and bounds resolved at store time, so a hit costs a
        // map probe, one epoch conversion, f64 comparisons, and two bounds-checked casts —
        // no file-record re-parse. The live summary is still re-read and must match the
        // memoized identity and bounds; any mismatch falls back to the full search below.
        if let Some(memo) = self.cache.spk_segment(id) {
            let et_s = epoch.to_et_seconds();
            // Tight f64 bounds: epochs at or beyond the segment edges generally miss the
            // memo and resolve through the full search's exact ±100 ns padded semantics.
            if et_s >= memo.start_et_s
                && et_s <= memo.end_et_s
                && let Some((_, spk)) = self.spk_data.get_index(memo.spk_no)
                && let Some(summaries) = spk.data_summaries_in_range(memo.summaries_byte_range)
                && let Some(summary) = summaries.get(memo.idx_in_spk)
                && summary.id() == id
                && summary.start_epoch_et_s() == memo.start_et_s
                && summary.end_epoch_et_s() == memo.end_et_s
            {
                return Ok((summary, memo.spk_no, memo.daf_idx, memo.idx_in_spk));
            }
        }

        for (rev_no, spk) in self.spk_data.values().rev().enumerate() {
            if let Ok((summary, daf_idx, idx_in_spk)) = spk.summary_from_id_at_epoch(id, epoch) {
                // NOTE: We're iterating backward, so the correct SPK number is "total loaded" minus "current iteration".
                let spk_no = self.num_loaded_spk() - rev_no - 1;
                // Memoize only when no other kernel can shadow this ID: the match must come
                // from the chronologically ordered index (unique segment per epoch within this
                // file), and no higher-precedence SPK may contain this ID at any epoch.
                // Ineligible IDs simply run the full search on every query.
                let memo_safe = spk.index.contains_key(&id)
                    && self
                        .spk_data
                        .values()
                        .rev()
                        .take(rev_no)
                        .all(|higher| higher.summary_from_id(id).is_err());
                if memo_safe && let Ok(range) = spk.summaries_byte_range(daf_idx) {
                    self.cache.store_spk_segment(
                        id,
                        SpkSegmentMemo {
                            spk_no,
                            daf_idx,
                            idx_in_spk,
                            summaries_byte_range: range,
                            start_et_s: summary.start_epoch_et_s(),
                            end_et_s: summary.end_epoch_et_s(),
                            decoded: None,
                        },
                    );
                }
                return Ok((summary, spk_no, daf_idx, idx_in_spk));
            }
        }

        // If the ID is not present at all, spk_domain will report it.
        let (start, end) = self.spk_domain(id)?;
        error!("Almanac: summary {id} valid from {start} to {end} but not at requested {epoch}");
        // If we're reached this point, there is no relevant summary at this epoch.
        Err(EphemerisError::SPK {
            action: "searching for SPK summary",
            source: DAFError::SummaryIdAtEpochError {
                kind: "SPK",
                id,
                epoch,
                start,
                end,
            },
        })
    }

    /// Returns the most recently loaded summary by its name, if any with that ID are available
    pub fn spk_summary_from_name(
        &self,
        name: &str,
    ) -> Result<(&SPKSummaryRecord, usize, Option<usize>, usize), EphemerisError> {
        for (spk_no, spk) in self.spk_data.values().rev().enumerate() {
            if let Ok((summary, daf_idx, idx_in_spk)) = spk.summary_from_name(name) {
                // NOTE: We're iterating backward, so the correct SPK number is "total loaded" minus "current iteration".
                return Ok((
                    summary,
                    self.num_loaded_spk() - spk_no - 1,
                    daf_idx,
                    idx_in_spk,
                ));
            }
        }

        // If we're reached this point, there is no relevant summary at this epoch.
        error!("Almanac: No summary {name} valid");

        Err(EphemerisError::SPK {
            action: "searching for SPK summary",
            source: DAFError::SummaryNameError {
                kind: "SPK",
                name: name.to_string(),
            },
        })
    }

    /// Returns the most recently loaded summary by its ID, if any with that ID are available
    pub fn spk_summary(
        &self,
        id: i32,
    ) -> Result<(&SPKSummaryRecord, usize, Option<usize>, usize), EphemerisError> {
        for (spk_no, spk) in self.spk_data.values().rev().enumerate() {
            if let Ok((summary, daf_idx, idx_in_spk)) = spk.summary_from_id(id) {
                // NOTE: We're iterating backward, so the correct SPK number is "total loaded" minus "current iteration".
                return Ok((
                    summary,
                    self.num_loaded_spk() - spk_no - 1,
                    daf_idx,
                    idx_in_spk,
                ));
            }
        }

        error!("Almanac: No summary {id} valid");
        // If we're reached this point, there is no relevant summary
        Err(EphemerisError::SPK {
            action: "searching for SPK summary",
            source: DAFError::SummaryIdError { kind: "SPK", id },
        })
    }
}

#[cfg_attr(feature = "python", pymethods)]
impl Almanac {
    /// Returns a vector of the summaries whose ID matches the desired `id`, in the order in which they will be used, i.e. in reverse loading order.
    ///
    /// # Warning
    /// This function performs a memory allocation.
    ///
    /// :type id: int
    /// :rtype: typing.List
    pub fn spk_summaries(&self, id: NaifId) -> Result<Vec<SPKSummaryRecord>, EphemerisError> {
        let mut summaries = vec![];
        for spk in self.spk_data.values().rev() {
            for these_summaries in spk.iter_summary_blocks().flatten() {
                for summary in these_summaries {
                    if summary.id() == id {
                        summaries.push(*summary);
                    }
                }
            }
        }

        if summaries.is_empty() {
            error!("Almanac: No summary {id} valid");
            // If we're reached this point, there is no relevant summary
            Err(EphemerisError::SPK {
                action: "searching for SPK summary",
                source: DAFError::SummaryIdError { kind: "SPK", id },
            })
        } else {
            Ok(summaries)
        }
    }

    /// Returns the applicable domain of the request id, i.e. start and end epoch that the provided id has loaded data.
    ///
    /// :type id: int
    /// :rtype: typing.Tuple
    pub fn spk_domain(&self, id: NaifId) -> Result<(Epoch, Epoch), EphemerisError> {
        let summaries = self.spk_summaries(id)?;

        // We know that the summaries is non-empty because if it is, the previous function call returns an error.
        let start = summaries
            .iter()
            .min_by_key(|summary| summary.start_epoch())
            .expect("summaries is non-empty, guaranteed by spk_summaries")
            .start_epoch();

        let end = summaries
            .iter()
            .max_by_key(|summary| summary.end_epoch())
            .expect("summaries is non-empty, guaranteed by spk_summaries")
            .end_epoch();

        Ok((start, end))
    }

    /// Returns a map of each loaded SPK ID to its domain validity.
    ///
    /// # Warning
    /// This function performs a memory allocation.
    ///
    /// :rtype: typing.Dict
    pub fn spk_domains(&self) -> Result<HashMap<NaifId, (Epoch, Epoch)>, EphemerisError> {
        ensure!(self.num_loaded_spk() > 0, NoEphemerisLoadedSnafu);

        let mut domains = HashMap::new();
        for spk in self.spk_data.values().rev() {
            for these_summaries in spk.iter_summary_blocks().flatten() {
                for summary in these_summaries {
                    let this_id = summary.id();
                    match domains.get_mut(&this_id) {
                        Some((cur_start, cur_end)) => {
                            if *cur_start > summary.start_epoch() {
                                *cur_start = summary.start_epoch();
                            }
                            if *cur_end < summary.end_epoch() {
                                *cur_end = summary.end_epoch();
                            }
                        }
                        None => {
                            domains.insert(this_id, (summary.start_epoch(), summary.end_epoch()));
                        }
                    }
                }
            }
        }

        Ok(domains)
    }
}

#[cfg(test)]
mod ut_almanac_spk {
    use crate::{
        constants::frames::{EARTH_J2000, MOON_J2000},
        prelude::{Almanac, Epoch},
    };

    #[test]
    fn summaries_nothing_loaded() {
        let almanac = Almanac::default();
        let e = Epoch::now().unwrap();

        assert!(
            almanac.spk_summary(0).is_err(),
            "empty Almanac should report an error"
        );
        assert!(
            almanac.spk_summary_at_epoch(0, e).is_err(),
            "empty Almanac should report an error"
        );
        assert!(
            almanac.spk_summary_from_name("invalid name").is_err(),
            "empty Almanac should report an error"
        );
        assert!(
            almanac
                .spk_summary_from_name_at_epoch("invalid name", e)
                .is_err(),
            "empty Almanac should report an error"
        );
    }

    #[test]
    fn queries_nothing_loaded() {
        let almanac = Almanac::default();
        let e = Epoch::now().unwrap();

        assert!(
            almanac.try_find_ephemeris_root().is_err(),
            "empty Almanac should report an error"
        );

        assert!(
            almanac.ephemeris_path_to_root(MOON_J2000, e).is_err(),
            "empty Almanac should report an error"
        );

        assert!(
            almanac
                .common_ephemeris_path(MOON_J2000, EARTH_J2000, e)
                .is_err(),
            "empty Almanac should report an error"
        );
    }

    #[test]
    fn cache_root_memo_and_invalidation() {
        let almanac = Almanac::new("../data/de440s.bsp").unwrap();
        assert!(almanac.cache.root().is_none(), "cache must start cold");

        let root = almanac.try_find_ephemeris_root().unwrap();
        assert_eq!(
            almanac.cache.root(),
            Some(root),
            "root must be memoized after the first call"
        );
        // Second call must return the memoized value.
        assert_eq!(almanac.try_find_ephemeris_root().unwrap(), root);

        // The segment memo must populate on a summary query and clear on invalidation.
        let epoch = Epoch::from_gregorian_at_noon(2024, 1, 1, hifitime::TimeScale::ET);
        almanac.spk_summary_at_epoch(301, epoch).unwrap();
        assert!(
            almanac.cache.spk_segment(301).is_some(),
            "segment memo must be populated after a query"
        );

        // Loading another SPK must invalidate the memo.
        let bytes = crate::file2heap!("../data/de440s.bsp").unwrap();
        let spk = crate::naif::SPK::parse(bytes).unwrap();
        let almanac = almanac.with_spk_as(spk, Some("copy".to_string()));
        assert!(
            almanac.cache.root().is_none(),
            "with_spk_as must invalidate the cache"
        );
        assert!(
            almanac.cache.spk_segment(301).is_none(),
            "with_spk_as must clear segment memos too"
        );
    }
}
