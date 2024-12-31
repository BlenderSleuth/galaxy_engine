// Copyright (c) 2024-2025 Ben Sutherland.

pub struct ScopeGuard<F: FnOnce()> {
    f: Option<F>,
}

impl<F: FnOnce()> ScopeGuard<F> {
    pub fn new(f: F) -> Self {
        Self { f: Some(f) }
    }
    pub fn defuse(&mut self) {
        self.f = None;
    }
}

impl<F: FnOnce()> Drop for ScopeGuard<F> {
    fn drop(&mut self) {
        if let Some(f) = self.f.take() {
            f();
        }
    }
}
