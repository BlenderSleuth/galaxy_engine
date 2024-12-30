// Copyright (c) 2024 Ben Sutherland.

use std::cell::RefCell;
use std::fs::File;
use std::io;
use std::io::Read;
use std::path::Path;

use bump_scope::{Bump, BumpScope, BumpString, BumpVec};
use itertools::Either;

// Bit of an experiment with arena allocators. Might be a bit silly.
thread_local! {
    pub static THREAD_LOCAL_ARENA_A: RefCell<Bump> = RefCell::new(Bump::new());
    pub static THREAD_LOCAL_ARENA_B: RefCell<Bump> = RefCell::new(Bump::new());
}

pub fn borrow_arenas<R, F: Fn(&mut BumpScope, &mut BumpScope) -> R>(f: F) -> R {
    THREAD_LOCAL_ARENA_A.with_borrow_mut(|arena_a| {
        THREAD_LOCAL_ARENA_B.with_borrow_mut(|arena_b| f(arena_a.as_mut_scope(), arena_b.as_mut_scope()))
    })
}

pub fn read_to_arena_string<'a, P: AsRef<Path>>(
    path: P,
    arena: &'a BumpScope,
) -> io::Result<BumpString<&'a BumpScope<'a>>> {
    fn inner<'a>(path: &Path, arena: &'a BumpScope) -> io::Result<BumpString<&'a BumpScope<'a>>> {
        let mut file = File::open(path)?;
        let size = file.metadata().map(|m| m.len() as usize).ok();
        let mut file_buf = BumpVec::new_in(arena);
        file_buf.resize_zeroed(size.unwrap_or(0));
        let mut cursor = 0;
        loop {
            let bytes_read = file.read(&mut file_buf[cursor..])?;
            if bytes_read == 0 {
                break;
            }
            cursor += bytes_read;
            file_buf.resize(file_buf.len() * 2, 0);
        }
        file_buf.truncate(cursor);
        let string = BumpString::from_utf8(file_buf).map_err(|_| io::ErrorKind::InvalidData)?;
        Ok(string)
    }
    inner(path.as_ref(), arena)
}

pub trait AllocIterator: Iterator {
    fn arena_partition_map<'a, 'b, F, L, R>(
        self,
        alloc_a: &'a BumpScope<'a>,
        alloc_b: &'b BumpScope<'b>,
        mut predicate: F,
    ) -> (BumpVec<L, &'a BumpScope<'a>>, BumpVec<R, &'b BumpScope<'b>>)
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Either<L, R>,
    {
        let mut left = BumpVec::new_in(alloc_a);
        let mut right = BumpVec::new_in(alloc_b);

        self.for_each(|val| match predicate(val) {
            Either::Left(v) => left.extend(Some(v)),
            Either::Right(v) => right.extend(Some(v)),
        });

        (left, right)
    }
}

impl<T> AllocIterator for T where T: Iterator + ?Sized {}
