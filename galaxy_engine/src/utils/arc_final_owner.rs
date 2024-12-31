// Copyright (c) 2024-2025 Ben Sutherland.

use std::mem::ManuallyDrop;
use std::sync::Arc;

// Used to allow sharing of objects, but also ensuring that it is destroyed at the appropriate time.
#[repr(transparent)]
pub struct ArcFinalOwner<T>(ManuallyDrop<Arc<T>>);

#[derive(Debug)]
pub enum FinalOwnerError {
    NotLastOwner,
}

impl<T> ArcFinalOwner<T> {
    pub fn new(value: T) -> Self {
        Self(ManuallyDrop::new(Arc::new(value)))
    }

    pub unsafe fn destroy_as_final(&mut self, destroy: impl FnOnce(&mut T)) -> Result<(), FinalOwnerError> {
        // Get shared item and drop it. Ensure we are the last owner of the shared reference.
        let object = unsafe { ManuallyDrop::take(&mut self.0) };
        match Arc::try_unwrap(object) {
            Ok(mut object) => {
                destroy(&mut object);
                Ok(())
            }
            Err(arc) => {
                log::error!("Not last owner of Vulkan object.");
                self.0 = ManuallyDrop::new(arc);
                Err(FinalOwnerError::NotLastOwner)
            }
        }
    }

    // Don't want to clone into another ArcFinalOwner, instead clone the Arc.
    pub fn clone(&self) -> Arc<T> {
        Arc::clone(&self.0)
    }
}

impl<T> AsRef<T> for ArcFinalOwner<T> {
    fn as_ref(&self) -> &T {
        &**self.0
    }
}

impl<T> std::ops::Deref for ArcFinalOwner<T> {
    type Target = Arc<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
