use std::{alloc::Layout, marker::PhantomData, mem::size_of, ops::Deref, ptr::NonNull};

use crate::mem::{copy_nonoverlapping, typed_copy_nonoverlapping};

/// A value that has no compile-type knonwn type information.
pub trait TypeErasedValue {
    /// Retrieves the number of bytes this value takes up in memory.
    fn size(&self) -> usize;

    /// Moves this value into the given `dst` pointer.
    ///
    /// # Safety
    /// The caller must guarantee that the `dst` pointer has enough memory allocate
    /// to store this value and can be interpreted as a pointer of the same type as the
    /// underlying value.
    unsafe fn move_into(self, dst: *mut u8);
}

/// A value that has its type known at compile-time.
pub struct TypedValue<T> {
    value: T,
}

impl<T> TypedValue<T> {
    /// Wraps the given `value` into a [TypedValue] struct.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> TypeErasedValue for TypedValue<T> {
    fn size(&self) -> usize {
        size_of::<T>()
    }

    #[inline]
    unsafe fn move_into(self, dst: *mut u8) {
        // SAFETY: Caller guarantees that this is safe.
        let dst = dst as *mut T;
        typed_copy_nonoverlapping(dst, &self.value as *const T);
        std::mem::forget(self.value);
    }
}

impl<T> Deref for TypedValue<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// An immutable reference to a type-erased value.
pub struct TypeErasedValueRef<'a> {
    ptr: *const u8,
    _phantom: PhantomData<&'a u8>,
}

impl<'a> TypeErasedValueRef<'a> {
    /// Creates a new type-erased value reference that uses the value pointed at by
    /// the given `ptr`.
    ///
    /// This references is guaranteed to always be valid since it "attaches" itself
    /// to the collection it was retrieved from. Meaning that as long as the collection
    /// exists this reference is valid, and since it is attached as an immutable reference
    /// no (re)allocation of the vector's buffer will be allowed.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given `ptr` is not null and points to a valid address
    /// in the collection the value exists in.
    #[inline]
    pub unsafe fn new_unchecked(ptr: *const u8) -> Self {
        debug_assert!(!ptr.is_null());

        Self {
            ptr,
            _phantom: PhantomData,
        }
    }

    /// Interprets this value as an immutable reference of the specified type.
    ///
    /// # Safety
    ///
    /// The caller must ensure that this value can be interpreted as the specified type.
    pub unsafe fn get<T>(&self) -> &T {
        let ptr = self.ptr as *const T;

        // SAFETY: Caller already ensured that `ptr` is not null.
        ptr.as_ref().unwrap()
    }
}

/// A mutable reference to a type-erased value.
pub struct TypeErasedValueMut<'a> {
    ptr: *mut u8,
    _phantom: PhantomData<&'a mut u8>,
}

impl<'a> TypeErasedValueMut<'a> {
    /// Creates a new type-erased value mutable reference that uses the value pointed
    /// at by the given `ptr`.
    ///
    /// This references is guaranteed to always be valid since it "attaches" itself
    /// to the collection it was retrieved from. Meaning that as long as the collection
    /// exists this reference is valid, and since it is attached as a mutable reference
    /// no (re)allocation of the vector's buffer will be allowed.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given `ptr` is not null and points to a valid address
    /// in the collection the value exists in.
    #[inline]
    pub unsafe fn new_unchecked(ptr: *mut u8) -> Self {
        debug_assert!(!ptr.is_null());

        Self {
            ptr,
            _phantom: PhantomData,
        }
    }

    /// Interprets this value as a mutable reference of the specified type.
    ///
    /// # Safety
    ///
    /// The caller must ensure that this value can be interpreted as the specified type.
    pub unsafe fn get_mut<T>(&self) -> &mut T {
        let ptr = self.ptr as *mut T;

        // SAFETY: Caller already ensured that `ptr` is not null.
        ptr.as_mut().unwrap()
    }
}

/// A value that owns its underlying memory.
///
/// Effectively a type-erased smart pointer.
/// This value allocates its own memory and deallocates it when dropped.
pub struct Value {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl Value {
    /// Constructs a new [Value] by creating a copy of the value
    /// pointed at by the given `src` pointer.
    ///
    /// The `layout` is used to allocate the new memory to store the
    /// copied value.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `layout.size() > 0`.
    /// The caller must ensure that `layout.size()` bytes can be copied
    /// from the given `src` pointer to the newly allocated memory.
    ///
    /// # Aborts
    ///
    /// Aborts if the memory allocation fails.
    #[inline]
    pub unsafe fn new_unchecked(src: *const u8, layout: Layout) -> Self {
        // SAFETY: Caller ensures that `layout.size() > 0`.
        let ptr = std::alloc::alloc(layout);
        let dst = match NonNull::new(ptr) {
            Some(x) => x,
            None => std::alloc::handle_alloc_error(layout),
        };

        // SAFETY: Caller ensures that this copy operation is safe
        copy_nonoverlapping(dst.as_ptr(), src, layout.size());

        Self { ptr: dst, layout }
    }

    /// Interprets this value as an immutable reference to the specified type.
    ///
    /// # Safety
    ///
    /// The caller must ensure that this value can be interpreted as the specified type.
    pub unsafe fn get<T>(&self) -> &T {
        self.ptr.cast::<T>().as_ref()
    }

    /// Interprets this value as a mutable reference to the specified type.
    ///
    /// # Safety
    ///
    /// The caller must ensure that this value can be interpreted as the specified type.
    pub unsafe fn get_mut<T>(&mut self) -> &mut T {
        self.ptr.cast::<T>().as_mut()
    }
}

impl TypeErasedValue for Value {
    #[inline]
    fn size(&self) -> usize {
        self.layout.size()
    }

    unsafe fn move_into(self, dst: *mut u8) {
        copy_nonoverlapping(dst, self.ptr.as_ptr(), self.size());
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: The `layout` is the same layout used when allocating the memory.
            std::alloc::dealloc(self.ptr.as_ptr(), self.layout)
        }
    }
}
