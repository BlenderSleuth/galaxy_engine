// Copyright (c) 2024 Ben Sutherland.

use std::collections::hash_map::Entry;

pub trait EntryExt<'a, V> {
    // Fallible version of Entry::or_insert_with.
    fn try_or_insert_with<E, F: FnOnce() -> Result<V, E>>(self, default: F) -> Result<&'a mut V, E>;
}

impl<'a, K, V> EntryExt<'a, V> for Entry<'a, K, V> {
    fn try_or_insert_with<E, F: FnOnce() -> Result<V, E>>(self, default: F) -> Result<&'a mut V, E> {
        Ok(match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()?),
        })
    }
}
