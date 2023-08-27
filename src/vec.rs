use std::{
    alloc::Layout,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::mem::array_layout;

use crate::value::{TypeErasedValue, TypeErasedValueMut, TypeErasedValueRef, Value};

/// Creates a [`TypeErasedVec`] containing the arguments.
///
/// `typed_erased_vec!` allows `TypeErasedVec`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`TypeErasedVec`] containing a given list of elements:
///
/// ```
/// let v = vecx::type_erased_vec![i32, 1, 2, 3];
/// unsafe {
///     assert_eq!(*v.get_unchecked(0).get::<i32>(), 1);
///     assert_eq!(*v.get_unchecked(1).get::<i32>(), 2);
///     assert_eq!(*v.get_unchecked(2).get::<i32>(), 3);
/// }
/// ```
///
/// - Create a [`TypeErasedVec`] from a given element and size:
///
/// ```
/// let v = vecx::type_erased_vec![i32, 1; 3];
/// assert_eq!(3, v.len());
/// unsafe {
///     assert_eq!(*v.get_unchecked(0).get::<i32>(), 1);
///     assert_eq!(*v.get_unchecked(1).get::<i32>(), 1);
///     assert_eq!(*v.get_unchecked(2).get::<i32>(), 1);
/// }
/// ```
///
/// Note that unlike array expressions this syntax supports all elements
/// which implement [`Clone`] and the number of elements doesn't have to be
/// a constant.
///
/// This will use `clone` to duplicate an expression, so one should be careful
/// using this with types having a nonstandard `Clone` implementation. For
/// example, `typed_erased_vec![Rc::new(1); 5]` will create a vector of five references
/// to the same boxed integer value, not five references pointing to independently
/// boxed integers.
///
/// Also, note that `typed_erased_vec![<type>, expr; 0]` is allowed, and produces an empty vector.
/// This will still evaluate `expr`, however, and immediately drop the resulting value, so
/// be mindful of side effects.
///
/// [`TypeErasedVec`]: vecx::TypeErasedVec
#[macro_export]
macro_rules! type_erased_vec {
    ($kind:ty) => {
        $crate::TypeErasedVec::new::<$kind>()
    };

    ($kind:ty, $elem:expr; $n:expr) => {
        {
            let mut vec = $crate::TypeErasedVec::new::<$kind>();
            vec.reserve_exact($n);

            for _ in 0..$n {
                unsafe {
                    let value = $crate::TypedValue::new($elem.clone());
                    vec.push(value);
                }
            }
            vec
        }
    };

    ($kind:ty, $($x:expr),*) => (
        {
            let mut vec = $crate::TypeErasedVec::new::<$kind>();
            unsafe {
                $(
                    let value = $crate::TypedValue::new($x);
                    vec.push(value);
                )*
            }
            vec
        }
    );
}

/// Function used to drop a single element in a [TypeErasedVec], pointed to by `data`.
pub type DropFn = unsafe fn(data: *mut u8);

/// A memory contiguous, type-erased dynamic array.
pub struct TypeErasedVec {
    element_layout: Layout,
    element_dropfn: Option<DropFn>,
    ptr: NonNull<u8>,
    len: usize,
    cap: usize,
}

impl TypeErasedVec {
    /// Constructs a new, empty `TypeErasedVec`.
    ///
    /// The `element_layout` describes the memory layout of each element in
    /// the vector and the `dropfn` can be used to specify the function to be
    /// called when an element is dropped.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecx::TypeErasedVec;
    /// use std::alloc::Layout;
    ///
    /// let mut vec: TypeErasedVec = TypeErasedVec::empty(Layout::new::<i32>(), None);
    /// ```
    #[inline]
    #[must_use]
    pub const fn empty(element_layout: Layout, dropfn: Option<DropFn>) -> Self {
        Self {
            element_layout,
            element_dropfn: dropfn,
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }

    /// Constructs a new, empty `TypeErasedVec` of type `T`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecx::TypeErasedVec;
    ///
    /// let mut vec: TypeErasedVec = TypeErasedVec::new::<i32>();
    /// ```
    #[inline]
    #[must_use]
    pub const fn new<T>() -> Self {
        Self {
            element_layout: Layout::new::<T>(),
            element_dropfn: if std::mem::needs_drop::<T>() {
                Some(|ptr| unsafe {
                    std::ptr::drop_in_place(ptr as *mut T);
                })
            } else {
                None
            },
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }

    /// Constructs a new, empty `TypeErasedVec` with at least the specified capacity.
    ///
    /// The `element_layout` describes the memory layout of each element in
    /// the vector and the `dropfn` can be used to specify the function to be
    /// called when an element is dropped.
    ///
    /// The vector will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is 0, the vector will not allocate.
    ///
    /// It is important to note that although the returned vector has the
    /// minimum *capacity* specified, the vector will have a zero *length*.
    ///
    /// If it is important to know the exact allocated capacity of a `TypeErasedVec`,
    /// always use the [`capacity`] method after construction.
    ///
    /// For `TypeErasedVec` where the elements are a zero-sized type, there will be no
    /// allocation and the capacity will always be `usize::MAX`.
    ///
    /// [`capacity`]: TypeErasedVec::capacity
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::alloc::Layout;
    /// use vecx::{TypeErasedVec, TypedValue};
    ///
    /// let mut vec = TypeErasedVec::with_capacity(Layout::new::<i32>(), None, 10);
    ///
    /// // The vector contains no items, even though it has capacity for more
    /// assert_eq!(vec.len(), 0);
    /// assert!(vec.capacity() >= 10);
    ///
    /// // These are all done without reallocating...
    /// for i in 0..10 {
    ///     unsafe {
    ///         vec.push(TypedValue::new(i));
    ///     }
    /// }
    /// assert_eq!(vec.len(), 10);
    /// assert!(vec.capacity() >= 10);
    ///
    /// // ...but this may make the vector reallocate
    /// unsafe { vec.push(TypedValue::new(11)) };
    /// assert_eq!(vec.len(), 11);
    /// assert!(vec.capacity() >= 11);
    ///
    /// // A vector of a zero-sized type will always over-allocate, since no
    /// // allocation is necessary
    /// let vec_units = TypeErasedVec::with_capacity(Layout::new::<()>(), None, 10);
    /// assert_eq!(vec_units.capacity(), usize::MAX);
    /// ```
    pub fn with_capacity(element_layout: Layout, dropfn: Option<DropFn>, capacity: usize) -> Self {
        if element_layout.size() == 0 || capacity == 0 {
            Self::empty(element_layout, dropfn)
        } else {
            let layout = match array_layout(element_layout, capacity) {
                Some(layout) => layout,
                None => capacity_overflow(),
            };

            match alloc_guard(layout.size()) {
                Ok(_) => {}
                Err(_) => capacity_overflow(),
            }

            let memory = unsafe {
                // SAFETY: At this point `layout.size() != 0`.
                std::alloc::alloc(layout)
            };

            Self {
                element_layout,
                element_dropfn: dropfn,
                ptr: match NonNull::new(memory) {
                    Some(ptr) => ptr,
                    None => std::alloc::handle_alloc_error(layout),
                },
                cap: capacity,
                len: 0,
            }
        }
    }
}

impl TypeErasedVec {
    /// Returns a raw pointer to the vector's buffer, or a dangling raw pointer
    /// valid for zero sized reads if the vector didn't allocate.
    ///
    /// The caller must ensure that the vector outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    /// Modifying the vector may cause its buffer to be reallocated,
    /// which would also make any pointers to it invalid.
    ///
    /// The caller must also ensure that the memory the pointer (non-transitively) points to
    /// is never written to (except inside an `UnsafeCell`) using this pointer or any pointer
    /// derived from it. If you need to mutate the contents of the slice, use [`as_mut_ptr`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::mem::size_of;
    ///
    /// let x = vecx::type_erased_vec![u32, 1, 2, 4];
    /// let x_ptr = x.as_ptr();
    ///
    /// unsafe {
    ///     for i in 0..x.len() {
    ///         assert_eq!(*x_ptr.add(i * size_of::<u32>()), 1 << i);
    ///     }
    /// }
    /// ```
    ///
    /// [`as_mut_ptr`]: TypeErasedVec::as_mut_ptr
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns an unsafe mutable pointer to the vector's buffer, or a dangling
    /// raw pointer valid for zero sized reads if the vector didn't allocate.
    ///
    /// The caller must ensure that the vector outlives the pointer this
    /// function returns, or else it will end up pointing to garbage.
    /// Modifying the vector may cause its buffer to be reallocated,
    /// which would also make any pointers to it invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::alloc::Layout;
    /// use vecx::TypeErasedVec;
    ///
    /// // Allocate vector big enough for 4 elements.
    /// let size = 4;
    /// let mut x = TypeErasedVec::with_capacity(Layout::new::<i32>(), None, size);
    /// let x_ptr = x.as_mut_ptr() as *mut i32;
    ///
    /// // Initialize elements via raw pointer writes, then set length.
    /// unsafe {
    ///     for i in 0..size {
    ///         *x_ptr.add(i) = i as i32;
    ///     }
    ///     x.set_len(size);
    ///
    ///     assert_eq!(*x.get_unchecked(0).get::<i32>(), 0);
    ///     assert_eq!(*x.get_unchecked(1).get::<i32>(), 1);
    ///     assert_eq!(*x.get_unchecked(2).get::<i32>(), 2);
    ///     assert_eq!(*x.get_unchecked(3).get::<i32>(), 3);
    /// }
    /// ```
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns the number of elements in the vector, also referred to
    /// as its 'length'.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = vecx::type_erased_vec![i32, 1, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the total number of elements the vector can hold without
    /// reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecx::{TypeErasedVec, TypedValue};
    /// use std::alloc::Layout;
    ///
    /// let mut vec = TypeErasedVec::with_capacity(Layout::new::<i32>(), None, 10);
    /// unsafe { vec.push(TypedValue::new(42)) };
    /// assert_eq!(vec.capacity(), 10);
    /// ```
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        if self.element_layout.size() == 0 {
            usize::MAX
        } else {
            self.cap
        }
    }

    /// Forces the length of the vector to `new_len`.
    ///
    /// This is a low-level operation that maintains none of the normal
    /// invariants of the type. Normally changing the length of a vector
    /// is done using one of the safe operations instead, such as
    /// [`truncate`], [`resize`], [`extend`], or [`clear`].
    ///
    /// [`truncate`]: TypeErased::truncate
    /// [`resize`]: TypeErased::resize
    /// [`extend`]: Extend::extend
    /// [`clear`]: TypeErased::clear
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`].
    /// - The elements at `old_len..new_len` must be initialized.
    ///
    /// [`capacity()`]: TypeErased::capacity
    ///
    /// # Examples
    ///
    /// This method can be useful for situations in which the vector
    /// is serving as a buffer for other code, particularly over FFI:
    ///
    /// ```no_run
    /// # #![allow(dead_code)]
    /// # // This is just a minimal skeleton for the doc example;
    /// # // don't use this as a starting point for a real library.
    /// # pub struct StreamWrapper { strm: *mut std::ffi::c_void }
    /// # const Z_OK: i32 = 0;
    /// # extern "C" {
    /// #     fn deflateGetDictionary(
    /// #         strm: *mut std::ffi::c_void,
    /// #         dictionary: *mut u8,
    /// #         dictLength: *mut usize,
    /// #     ) -> i32;
    /// # }
    /// # impl StreamWrapper {
    /// pub fn get_dictionary(&self) -> Option<vecx::TypeErasedVec> {
    ///     use std::alloc::Layout;
    ///
    ///     // Per the FFI method's docs, "32768 bytes is always enough".
    ///     let mut dict = vecx::TypeErasedVec::with_capacity(Layout::new::<u8>(), None, 32_768);
    ///     let mut dict_length = 0;
    ///     // SAFETY: When `deflateGetDictionary` returns `Z_OK`, it holds that:
    ///     // 1. `dict_length` elements were initialized.
    ///     // 2. `dict_length` <= the capacity (32_768)
    ///     // which makes `set_len` safe to call.
    ///     unsafe {
    ///         // Make the FFI call...
    ///         let r = deflateGetDictionary(self.strm, dict.as_mut_ptr(), &mut dict_length);
    ///         if r == Z_OK {
    ///             // ...and update the length to what was initialized.
    ///             dict.set_len(dict_length);
    ///             Some(dict)
    ///         } else {
    ///             None
    ///         }
    ///     }
    /// }
    /// # }
    /// ```
    ///
    /// While the following example is sound, there is a memory leak since
    /// the inner vectors were not freed prior to the `set_len` call:
    ///
    /// ```
    /// let mut vec = vecx::type_erased_vec![vecx::TypeErasedVec,
    ///     vecx::type_erased_vec![i32, 1, 0, 0],
    ///     vecx::type_erased_vec![i32, 0, 1, 0],
    ///     vecx::type_erased_vec![i32, 0, 0, 1]
    /// ];
    /// // SAFETY:
    /// // 1. `old_len..0` is empty so no elements need to be initialized.
    /// // 2. `0 <= capacity` always holds whatever `capacity` is.
    /// unsafe {
    ///     vec.set_len(0);
    /// }
    /// ```
    ///
    /// Normally, here, one would use [`clear`] instead to correctly drop
    /// the contents and thus not leak memory.
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());

        self.len = new_len;
    }

    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecx::{TypeErasedVec, TypedValue};
    ///
    /// let mut v = TypeErasedVec::new::<i32>();
    /// assert!(v.is_empty());
    ///
    /// unsafe { v.push(TypedValue::new(1)) };
    /// assert!(!v.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the vector, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vecx::type_erased_vec![i32, 1, 2, 3];
    ///
    /// v.clear();
    ///
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        if let Some(dropfn) = self.element_dropfn {
            let size = self.element_layout.size();

            for i in 0..self.len() {
                // Offset is valid since `i < len`.
                let offset = i * size;

                unsafe {
                    // SAFETY: `ptr` pointer is valid since all memory up to `len` is valid.
                    let ptr = self.ptr.as_ptr().add(offset);
                    (dropfn)(ptr);
                }
            }
        }

        self.len = 0;
    }

    /// Removes the last element from this vector and returns it, or [`None`] if it
    /// is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vecx::type_erased_vec![i32, 1, 2, 3];
    /// unsafe {
    ///     assert_eq!(*vec.pop().unwrap().get::<i32>(), 3);
    ///     assert_eq!(*vec.get_unchecked(0).get::<i32>(), 1);
    ///     assert_eq!(*vec.get_unchecked(1).get::<i32>(), 2);
    /// }
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<Value> {
        let len = self.len();
        if len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;

                let offset = self.len() * self.element_layout.size();
                let end = self.as_ptr().add(offset);

                // SAFETY: Offset will point to the last element in the vector,
                // and as such is a valid pointer. The constructed `OwnedValue`
                // will make a copy of the last element and as such the vector
                // can (de)allocate memory without interfering with this value.
                Some(Value::new_unchecked(end, self.element_layout))
            }
        }
    }

    /// Appends an element to the back of the vector.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given `value` matches the expected
    /// value type of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecx::TypedValue;
    ///
    /// let mut vec = vecx::type_erased_vec![i32, 1, 2];
    /// unsafe {
    ///     vec.push(TypedValue::new(3));
    ///     assert_eq!(*vec.get_unchecked(0).get::<i32>(), 1);
    ///     assert_eq!(*vec.get_unchecked(1).get::<i32>(), 2);
    ///     assert_eq!(*vec.get_unchecked(2).get::<i32>(), 3);
    /// }
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub unsafe fn push<V: TypeErasedValue>(&mut self, value: V) {
        debug_assert!(self.element_layout.size() == value.size());

        // This will panic or abort if we would allocate > isize::MAX bytes
        // or if the length increment would overflow for zero-sized types.
        let len = self.len();
        if len == self.capacity() {
            self.reserve_for_push(len);
        }

        unsafe {
            // SAFETY: The vector is guaranteeed to not allocate > isize::MAX bytes
            // so the computed offset is guaranteed to be < isize::MAX.
            let end = self.as_mut_ptr().add(len * self.element_layout.size());
            value.move_into(end);
            self.len += 1;
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecx::TypedValue;
    ///
    /// let mut vec = vecx::type_erased_vec![i32, 1, 2, 3];
    /// unsafe {
    ///     vec.insert(1, TypedValue::new(4));
    ///     assert_eq!(*vec.get_unchecked(0).get::<i32>(), 1);
    ///     assert_eq!(*vec.get_unchecked(1).get::<i32>(), 4);
    ///     assert_eq!(*vec.get_unchecked(2).get::<i32>(), 2);
    ///     assert_eq!(*vec.get_unchecked(3).get::<i32>(), 3);
    ///     vec.insert(4, TypedValue::new(5));
    ///     assert_eq!(*vec.get_unchecked(0).get::<i32>(), 1);
    ///     assert_eq!(*vec.get_unchecked(1).get::<i32>(), 4);
    ///     assert_eq!(*vec.get_unchecked(2).get::<i32>(), 2);
    ///     assert_eq!(*vec.get_unchecked(3).get::<i32>(), 3);
    ///     assert_eq!(*vec.get_unchecked(4).get::<i32>(), 5);
    /// }
    /// ```
    #[cfg(not(no_global_oom_handling))]
    pub unsafe fn insert<V: TypeErasedValue>(&mut self, index: usize, value: V) {
        debug_assert!(self.element_layout.size() == value.size());

        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("insertion index (is {index}) should be <= len (is {len})");
        }

        let len = self.len();
        let element_size = self.element_layout.size();

        // space for the new element
        if len == self.capacity() {
            self.reserve(1);
        }

        unsafe {
            // infallible
            // The spot to put the new value
            {
                let ptr = self.as_mut_ptr().add(index * element_size);
                if index < len {
                    // Shift everything over to make space. (Duplicating the
                    // `index`th element into two consecutive places.)
                    std::ptr::copy(ptr, ptr.add(element_size), (len - index) * element_size);
                } else if index == len {
                    // No elements need shifting.
                } else {
                    assert_failed(index, len);
                }

                // Write it in, overwriting the first copy of the `index`th element.
                value.move_into(ptr);
            }

            self.set_len(len + 1);
        }
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// Note: Because this shifts over the remaining elements, it has a
    /// worst-case performance of *O*(*n*). If you don't need the order of elements
    /// to be preserved, use [`swap_remove`] instead.
    ///
    /// [`swap_remove`]: TypeErasedVec::swap_remove
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vecx::type_erased_vec![i32, 1, 2, 3];
    /// unsafe {
    ///     assert_eq!(*v.remove(1).get::<i32>(), 2);
    ///     assert_eq!(*v.get_unchecked(0).get::<i32>(), 1);
    ///     assert_eq!(*v.get_unchecked(1).get::<i32>(), 3);
    /// }
    /// ```
    #[track_caller]
    pub fn remove(&mut self, index: usize) -> Value {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("removal index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        let element_size = self.element_layout.size();

        if index >= len {
            assert_failed(index, len);
        }

        unsafe {
            // infallible
            let value;
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().add(index * element_size);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                value = Value::new_unchecked(ptr, self.element_layout);

                // Shift everything down to fill in that spot.
                std::ptr::copy(ptr.add(element_size), ptr, (len - index - 1) * element_size);
            }

            self.set_len(len - 1);
            value
        }
    }

    /// Tries to removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// Unlike [`remove`] this function does not panic if `index` is out of bounds.
    ///
    /// Note: Because this shifts over the remaining elements, it has a
    /// worst-case performance of *O*(*n*). If you don't need the order of elements
    /// to be preserved, use [`swap_remove`] instead.
    ///
    /// [`remove`]: TypeErasedVec::remove
    /// [`swap_remove`]: TypeErasedVec::swap_remove
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vecx::type_erased_vec![i32, 1, 2, 3];
    /// unsafe {
    ///     assert_eq!(*v.try_remove(1).unwrap().get::<i32>(), 2);
    ///     assert!(v.try_remove(3).is_none());
    /// }
    /// ```
    #[track_caller]
    pub fn try_remove(&mut self, index: usize) -> Option<Value> {
        let len = self.len();
        let element_size = self.element_layout.size();

        if index >= len {
            return None;
        }

        unsafe {
            // infallible
            let value;
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().add(index * element_size);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                value = Value::new_unchecked(ptr, self.element_layout);

                // Shift everything down to fill in that spot.
                std::ptr::copy(ptr.add(element_size), ptr, (len - index - 1) * element_size);
            }

            self.set_len(len - 1);
            Some(value)
        }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// This does not preserve ordering, but is *O*(1).
    /// If you need to preserve the element order, use [`remove`] instead.
    ///
    /// [`remove`]: TypeErasedVec::remove
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vecx::type_erased_vec![i32, 1, 2, 3, 4];
    ///
    /// unsafe {
    ///     assert_eq!(*v.swap_remove(1).get::<i32>(), 2);
    ///     // [1, 4, 3]
    ///     assert_eq!(*v.get_unchecked(0).get::<i32>(), 1);
    ///     assert_eq!(*v.get_unchecked(1).get::<i32>(), 4);
    ///     assert_eq!(*v.get_unchecked(2).get::<i32>(), 3);
    ///
    ///     assert_eq!(*v.swap_remove(0).get::<i32>(), 1);
    ///     // [3, 4]
    ///     assert_eq!(*v.get_unchecked(0).get::<i32>(), 3);
    ///     assert_eq!(*v.get_unchecked(1).get::<i32>(), 4);
    /// }
    /// ```
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> Value {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("swap_remove index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        let element_size = self.element_layout.size();

        if index >= len {
            assert_failed(index, len);
        }

        unsafe {
            // the place we are taking from.
            let ptr = self.as_ptr().add(index * element_size);

            // We replace self[index] with the last element. Note that if the
            // bounds check above succeeds there must be a last element (which
            // can be self[index] itself).
            let value = Value::new_unchecked(ptr, self.element_layout);
            let base_ptr = self.as_mut_ptr();

            std::ptr::copy(
                base_ptr.add((len - 1) * element_size),
                base_ptr.add(index * element_size),
                element_size,
            );
            self.set_len(len - 1);
            value
        }
    }

    /// Removes the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// Note: Because this shifts over the remaining elements, it has a
    /// worst-case performance of *O*(*n*). If you don't need the order of elements
    /// to be preserved, use [`swap_remove`] instead.
    ///
    /// Unlike [`remove`] this function does not return the actual removed element, for
    /// performance reasons.
    ///
    /// [`remove`]: TypeErasedVec::remove
    /// [`swap_remove`]: TypeErasedVec::swap_remove
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vecx::type_erased_vec![i32, 1, 2, 3];
    /// v.purge(1);
    /// assert_eq!(v.len(), 2);
    /// assert_eq!(unsafe { *v.get_unchecked(1).get::<i32>() }, 3);
    ///
    /// v.purge(0);
    /// assert_eq!(v.len(), 1);
    /// assert_eq!(unsafe { *v.get_unchecked(0).get::<i32>() }, 3);
    /// ```
    #[track_caller]
    pub fn purge(&mut self, index: usize) {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("purge index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        let element_size = self.element_layout.size();

        if index >= len {
            assert_failed(index, len);
        }

        unsafe {
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().add(index * element_size);

                // Shift everything down to fill in that spot.
                std::ptr::copy(ptr.add(element_size), ptr, (len - index - 1) * element_size);
            }

            self.set_len(len - 1);
        }
    }

    /// Tries to remove the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// Returns whether an element was removed.
    ///
    /// Note: Because this shifts over the remaining elements, it has a
    /// worst-case performance of *O*(*n*). If you don't need the order of elements
    /// to be preserved, use [`swap_remove`] instead.
    ///
    /// Unlike [`remove`] this function does not return the actual removed element, for
    /// performance reasons.
    ///
    /// Unlike [`purge`] this function does not panic if `index` is out of bounds.
    ///
    /// [`remove`]: TypeErasedVec::remove
    /// [`purge`]: TypeErasedVec::purge
    /// [`swap_remove`]: TypeErasedVec::swap_remove
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vecx::type_erased_vec![i32, 1, 2, 3];
    /// assert!(v.try_purge(1));
    /// assert_eq!(v.len(), 2);
    /// assert_eq!(unsafe { *v.get_unchecked(1).get::<i32>() }, 3);
    ///
    /// assert!(!v.try_purge(2));
    /// assert_eq!(v.len(), 2);
    /// ```
    #[track_caller]
    pub fn try_purge(&mut self, index: usize) -> bool {
        let len = self.len();
        let element_size = self.element_layout.size();

        if index >= len {
            return false;
        }

        unsafe {
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().add(index * element_size);

                // Shift everything down to fill in that spot.
                std::ptr::copy(ptr.add(element_size), ptr, (len - index - 1) * element_size);
            }

            self.set_len(len - 1);
            true
        }
    }

    /// Removes an element from the vector without returning it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// This does not preserve ordering, but is *O*(1).
    /// If you need to preserve the element order, use [`purge`] instead.
    ///
    /// [`purge`]: TypeErasedVec::purge
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = vecx::type_erased_vec![i32, 1, 2, 3, 4];
    ///
    /// unsafe {
    ///     v.swap_purge(1);
    ///     assert_eq!(v.len(), 3);
    ///     // [1, 4, 3]
    ///     assert_eq!(*v.get_unchecked(0).get::<i32>(), 1);
    ///     assert_eq!(*v.get_unchecked(1).get::<i32>(), 4);
    ///     assert_eq!(*v.get_unchecked(2).get::<i32>(), 3);
    ///
    ///     v.swap_purge(0);
    ///     assert_eq!(v.len(), 2);
    ///     // [3, 4]
    ///     assert_eq!(*v.get_unchecked(0).get::<i32>(), 3);
    ///     assert_eq!(*v.get_unchecked(1).get::<i32>(), 4);
    /// }
    /// ```
    #[inline]
    pub fn swap_purge(&mut self, index: usize) {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("swap_purge index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        let element_size = self.element_layout.size();

        if index >= len {
            assert_failed(index, len);
        }

        unsafe {
            // We replace self[index] with the last element. Note that if the
            // bounds check above succeeds there must be a last element (which
            // can be self[index] itself).
            let base_ptr = self.as_mut_ptr();

            std::ptr::copy(
                base_ptr.add((len - 1) * element_size),
                base_ptr.add(index * element_size),
                element_size,
            );
            self.set_len(len - 1);
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in this vector. The collection may reserve more space to
    /// speculatively avoid frequent reallocations. After calling `reserve`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vecx::type_erased_vec![i32, 1];
    /// vec.reserve(10);
    /// assert!(vec.capacity() >= 11);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    pub fn reserve(&mut self, additional: usize) {
        // Callers expect this function to be very cheap when there is already sufficient capacity.
        // Therefore, we move all the resizing and error-handling logic from grow_amortized and
        // handle_reserve behind a call, while making sure that this function is likely to be
        // inlined as just a comparison and a call if the comparison fails.
        #[cold]
        fn do_reserve_and_handle(slf: &mut TypeErasedVec, len: usize, additional: usize) {
            handle_reserve(slf.grow_amortized(len, additional));
        }

        let len = self.len();
        if self.needs_to_grow(len, additional) {
            do_reserve_and_handle(self, len, additional);
        }
    }

    /// Reserves the minimum capacity for at least `additional` more elements to
    /// be inserted in the given `TypeErasedVec`. Unlike [`reserve`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    /// After calling `reserve_exact`, capacity will be greater than or equal to
    /// `self.len() + additional`. Does nothing if the capacity is already
    /// sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer [`reserve`] if future insertions are expected.
    ///
    /// [`reserve`]: TypeErased::reserve
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vecx::type_erased_vec![i32, 1];
    /// vec.reserve_exact(10);
    /// assert!(vec.capacity() >= 11);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    pub fn reserve_exact(&mut self, additional: usize) {
        handle_reserve(self.try_reserve_exact(self.len(), additional));
    }

    /// The same as `reserve_exact`, but returns on errors instead of panicking or aborting.
    pub fn try_reserve_exact(
        &mut self,
        len: usize,
        additional: usize,
    ) -> Result<(), TryReserveError> {
        if self.needs_to_grow(len, additional) {
            self.grow_exact(len, additional)
        } else {
            Ok(())
        }
    }

    /// Returns a reference to an element in the given `index` or `None` if
    /// out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = vecx::type_erased_vec![i32, 10, 40, 30];
    ///
    /// let value = v.get(1);
    /// unsafe {
    ///     assert!(value.is_some());
    ///
    ///     let value = value.unwrap();
    ///     assert_eq!(40, *value.get::<i32>());
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<TypeErasedValueRef> {
        if index >= self.len() {
            None
        } else {
            unsafe {
                // SAFETY: At this point `index < len`.
                Some(self.get_unchecked(index))
            }
        }
    }

    /// Returns a reference to an element in the given `index` without
    /// performing any bound checks.
    ///
    /// For a safe alternative see [`get`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// [`get`]: TypeErasedVec::get
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// let v = vecx::type_erased_vec![i32, 10, 40, 30];
    ///
    /// unsafe {
    ///     let value = v.get_unchecked(1);
    ///     assert_eq!(40, *value.get::<i32>());
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked(&self, index: usize) -> TypeErasedValueRef {
        // SAFETY: The `offset` is guaranteed by the caller to be less than `len`, and as such
        // the pointer `data + offset` will be a valid memory region.
        let offset = index * self.element_layout.size();

        // SAFETY: Caller ensures that `index < len`, as such the computed `offset` will
        // be `< len * size`, which itself is `< isize::MAX`, since the vector does not allocate
        // more than `isize::MAX` bytes.
        let ptr = self.ptr.as_ptr().add(offset);

        // SAFETY: Caller ensures that the `ptr` pointer is valid, and as such not null.
        TypeErasedValueRef::new_unchecked(ptr)
    }

    /// Returns a mutable reference to an element or `None` if
    /// the index is out of bounds.
    ///
    /// [`get`]: TypeErasedVec::get
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = vecx::type_erased_vec![i32, 0, 1, 2];
    ///
    /// if let Some(elem) = x.get_mut(1) {
    ///     unsafe {
    ///         let value = elem.get_mut::<i32>();
    ///         *value = 42;
    ///     }
    /// }
    ///
    /// unsafe {
    ///     let elem = x.get(1).unwrap();
    ///     let value = elem.get::<i32>();
    ///     assert_eq!(42, *value);
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<TypeErasedValueMut> {
        if index >= self.len() {
            None
        } else {
            unsafe {
                // SAFETY: At this point `index < len`.
                Some(self.get_unchecked_mut(index))
            }
        }
    }

    /// Returns a mutable reference to an element in the given `index` without
    /// performing any bound checks.
    ///
    /// For a safe alternative see [`get_mut`].
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// [`get_mut`]: TypeErasedVec::get_mut
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = vecx::type_erased_vec![i32, 1, 2, 4];
    ///
    /// unsafe {
    ///     let value = x.get_unchecked_mut(1);
    ///     assert_eq!(2, *value.get_mut::<i32>());
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> TypeErasedValueMut {
        // SAFETY: The `offset` is guaranteed by the caller to be less than `len`, and as such
        // the pointer `data + offset` will be a valid memory region.
        let offset = index * self.element_layout.size();

        // SAFETY: Caller ensures that `index < len`, as such the computed `offset` will
        // be `< len * size`, which itself is `< isize::MAX`, since the vector does not allocate
        // more than `isize::MAX` bytes.
        let ptr = self.ptr.as_ptr().add(offset);

        // SAFETY: Caller ensures that the `ptr` pointer is valid, and as such not null.
        TypeErasedValueMut::new_unchecked(ptr)
    }

    /// Returns an iterator over the vector.
    ///
    /// The iterator yields all items from start to end.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = vecx::type_erased_vec![i32, 1, 2, 4];
    /// let mut iterator = x.iter();
    ///
    /// unsafe {
    ///     assert_eq!(*iterator.next().unwrap().get::<i32>(), 1);
    ///     assert_eq!(*iterator.next().unwrap().get::<i32>(), 2);
    ///     assert_eq!(*iterator.next().unwrap().get::<i32>(), 4);
    ///     assert!(iterator.next().is_none());
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> TypeErasedIter {
        TypeErasedIter::new(self, self.element_layout.size())
    }

    /// Returns an iterator that allows modifying each value.
    ///
    /// The iterator yields all items from start to end.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = vecx::type_erased_vec![i32, 1, 2, 4];
    /// for element in x.iter_mut() {
    ///     let value = unsafe { element.get_mut::<i32>() };
    ///     *value += 2;
    /// }
    ///
    /// unsafe {
    ///     assert_eq!(*x.get_unchecked(0).get::<i32>(), 3);
    ///     assert_eq!(*x.get_unchecked(1).get::<i32>(), 4);
    ///     assert_eq!(*x.get_unchecked(2).get::<i32>(), 6);
    /// }
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> TypeErasedIterMut {
        let element_size = self.element_layout.size();
        TypeErasedIterMut::new(self, element_size)
    }
}

impl TypeErasedVec {
    /// A specialized version of `reserve()` used only by the hot and
    /// oft-instantiated `push()`, which does its own capacity check.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    fn reserve_for_push(&mut self, len: usize) {
        handle_reserve(self.grow_amortized(len, 1));
    }

    /// Returns if the buffer needs to grow to fulfill the needed extra capacity.
    /// Mainly used to make inlining reserve-calls possible without inlining `grow`.
    fn needs_to_grow(&self, len: usize, additional: usize) -> bool {
        additional > self.capacity().wrapping_sub(len)
    }

    /// This method is usually instantiated many times. So we want it to be as
    /// small as possible, to improve compile times. But we also want as much of
    /// its contents to be statically computable as possible, to make the
    /// generated code run faster. Therefore, this method is carefully written
    /// so that all of the code that depends on `T` is within it, while as much
    /// of the code that doesn't depend on `T` as possible is in functions that
    /// are non-generic over `T`.
    #[inline]
    fn grow_amortized(&mut self, len: usize, additional: usize) -> Result<(), TryReserveError> {
        // This is ensured by the calling contexts.
        debug_assert!(additional > 0);

        if self.element_layout.size() == 0 {
            // Since we return a capacity of `usize::MAX` when `elem_size` is
            // 0, getting to here necessarily means the `RawVec` is overfull.
            return Err(TryReserveError::CapacityOverflow);
        }

        // Nothing we can really do about these checks, sadly.
        let required_cap = len
            .checked_add(additional)
            .ok_or(TryReserveError::CapacityOverflow)?;

        // This guarantees exponential growth. The doubling cannot overflow
        // because `cap <= isize::MAX` and the type of `cap` is `usize`.
        let cap = std::cmp::max(self.cap * 2, required_cap);
        let cap = std::cmp::max(self.min_non_zero_cap(), cap);

        let new_layout = array_layout(self.element_layout, cap);
        self.ptr = finish_grow(new_layout, self.current_memory())?;
        self.cap = cap;

        Ok(())
    }

    // The constraints on this method are much the same as those on
    // `grow_amortized`, but this method is usually instantiated less often so
    // it's less critical.
    #[inline]
    fn grow_exact(&mut self, len: usize, additional: usize) -> Result<(), TryReserveError> {
        if self.element_layout.size() == 0 {
            // Since we return a capacity of `usize::MAX` when the type size is
            // 0, getting to here necessarily means the `RawVec` is overfull.
            return Err(TryReserveError::CapacityOverflow);
        }

        let cap = len
            .checked_add(additional)
            .ok_or(TryReserveError::CapacityOverflow)?;

        // `finish_grow` is non-generic over `T`.
        let new_layout = array_layout(self.element_layout, cap);
        self.ptr = finish_grow(new_layout, self.current_memory())?;
        self.cap = cap;

        Ok(())
    }

    /// Retrieves the current data pointer and layout of the memory of this vector.
    #[inline]
    fn current_memory(&self) -> Option<(NonNull<u8>, Layout)> {
        if self.element_layout.size() == 0 || self.cap == 0 {
            None
        } else {
            // We could use Layout::array here which ensures the absence of isize and usize overflows
            // and could hypothetically handle differences between stride and size, but this memory
            // has already been allocated so we know it can't overflow and currently rust does not
            // support such types. So we can do better by skipping some checks and avoid an unwrap.
            debug_assert!(self.element_layout.size() % self.element_layout.align() == 0);

            unsafe {
                let align = self.element_layout.align();
                // Unstable: unchecked_mul
                let size = self.element_layout.size() * self.cap;
                let layout = Layout::from_size_align_unchecked(size, align);
                Some((self.ptr, layout))
            }
        }
    }

    /// Retrieves minimum (non-zero) capacity of this vector.
    #[inline]
    const fn min_non_zero_cap(&self) -> usize {
        let element_size = self.element_layout.size();
        if element_size == 1 {
            8
        } else if element_size <= 1024 {
            4
        } else {
            1
        }
    }
}

impl Deref for TypeErasedVec {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe {
            // SAFETY: TypeErasedVec guarantees that `data` is valid trough `len` elements.
            std::slice::from_raw_parts(self.as_ptr(), self.len())
        }
    }
}

impl DerefMut for TypeErasedVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            // SAFETY: TypeErasedVec guarantees that `data` is valid trough `len` elements.
            std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len())
        }
    }
}

impl Drop for TypeErasedVec {
    fn drop(&mut self) {
        // Also sets the length to zero.
        self.clear();

        if let Some((ptr, layout)) = self.current_memory() {
            unsafe {
                // SAFETY: `ptr` was allocated using the same (Global) allocator being
                // used to deallocate the memory and `layout` was the previous layout
                // used to allocate the current memory region of the vector.
                std::alloc::dealloc(ptr.as_ptr(), layout);
            }
        }
    }
}

/// Immutable type-erased value iterator.
///
/// This struct is created by the [`iter`] method.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let vec = vecx::type_erased_vec![i32, 1, 2, 3];
///
/// for element in vec.iter() {
///     println!("{}", unsafe { *element.get::<i32>() });
/// }
/// ```
///
/// [`iter`]: TypeErased::iter
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct TypeErasedIter<'a> {
    /// The pointer to the next element to return, or the past-the-end location
    /// if the iterator is empty.
    ///
    /// This address will be used for all ZST elements, never changed.
    ptr: NonNull<u8>,
    // If the elements are a ZST, this is actually ptr+len. This encoding is picked so that
    // ptr == end is a quick test for the Iterator being empty, that works
    // for both ZST and non-ZST.
    end: *const u8,
    /// Size, in bytes, of the elements being iterated.
    element_size: usize,
    _marker: PhantomData<&'a u8>,
}

impl<'a> TypeErasedIter<'a> {
    /// Constructs a new type-erased immutable iterator.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pointer past-the-end element
    /// is valid, for elements of `element_size`.
    #[inline]
    fn new(slice: &'a [u8], element_size: usize) -> Self {
        let ptr = slice.as_ptr();
        unsafe {
            let end = if element_size == 0 {
                ptr.wrapping_add(slice.len())
            } else {
                // SAFETY: Caller ensures that this is valid.
                ptr.add(slice.len() * element_size)
            };

            Self {
                ptr: NonNull::new_unchecked(ptr as *mut _),
                end,
                element_size,
                _marker: PhantomData,
            }
        }
    }
}

impl<'a> Iterator for TypeErasedIter<'a> {
    type Item = TypeErasedValueRef<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.as_ptr() as *const u8 == self.end {
            // Reach end of iterator
            return None;
        }

        unsafe {
            if self.element_size == 0 {
                // SAFETY: Caller guarantees that `ptr` is always valid
                let value = TypeErasedValueRef::new_unchecked(self.ptr.as_ptr());
                // For ZST the number of elements in this iterator is the kept as the difference in bytes
                // between the `end` and `ptr` pointers. So everytime we read a ZST element we bring
                // the `end` pointer one step closer to the `ptr` pointer.
                self.end = self.end.wrapping_sub(1);
                return Some(value);
            }

            // SAFETY: Caller guarantees that `ptr` is always valid
            let value = TypeErasedValueRef::new_unchecked(self.ptr.as_ptr());
            let ptr = self.ptr.as_ptr().add(self.element_size);
            self.ptr = NonNull::new_unchecked(ptr);
            Some(value)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.len();
        (exact, Some(exact))
    }
}

impl<'a> ExactSizeIterator for TypeErasedIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        if self.element_size == 0 {
            unsafe { self.end.offset_from(self.ptr.as_ptr()) as usize }
        } else {
            // The pointers are expressed in bytes, and as such the offset difference needs to be divided
            // by the size of each element so we are left with the number of elements (not the number of bytes).
            unsafe { self.end.offset_from(self.ptr.as_ptr()) as usize / self.element_size }
        }
    }
}

/// Mutable type-erased iterator.
///
/// This struct is created by the [`iter_mut`] method.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // First, we declare a type which has `iter_mut` method to get the `IterMut`
/// // struct (`&[usize]` here):
/// let mut vec = vecx::type_erased_vec![i32, 1, 2, 3];
///
/// // Then, we iterate over it and increment each element value:
/// for element in vec.iter_mut() {
///     let value = unsafe { element.get_mut::<i32>() };
///     *value += 1;
/// }
///
/// // We now have "[2, 3, 4]":
/// for element in vec.iter() {
///     println!("{}", unsafe { *element.get::<i32>() });
/// }
/// ```
///
/// [`iter_mut`]: TypeErasedVec::iter_mut
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct TypeErasedIterMut<'a> {
    /// The pointer to the next element to return, or the past-the-end location
    /// if the iterator is empty.
    ///
    /// This address will be used for all ZST elements, never changed.
    ptr: NonNull<u8>,
    // If the elements are a ZST, this is actually ptr+len. This encoding is picked so that
    // ptr == end is a quick test for the Iterator being empty, that works
    // for both ZST and non-ZST.
    end: *mut u8,
    /// Size, in bytes, of the elements being iterated.
    element_size: usize,
    _marker: PhantomData<&'a mut u8>,
}

impl<'a> TypeErasedIterMut<'a> {
    /// Constructs a new type-erased immutable iterator.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pointer past-the-end element
    /// is valid, for elements of `element_size`.
    #[inline]
    fn new(slice: &'a mut [u8], element_size: usize) -> Self {
        let ptr = slice.as_mut_ptr();
        // SAFETY: There are several things here:
        //
        // `ptr` has been obtained by `slice.as_ptr()` where `slice` is a valid
        // reference thus it is non-NULL and safe to use and pass to
        // `NonNull::new_unchecked` .
        //
        // Adding `slice.len()` to the starting pointer gives a pointer
        // at the end of `slice`. `end` will never be dereferenced, only checked
        // for direct pointer equality with `ptr` to check if the iterator is
        // done.
        //
        // In the case of a ZST, the end pointer is just the start pointer plus
        // the length, to also allows for the fast `ptr == end` check.
        unsafe {
            let end = if element_size == 0 {
                ptr.wrapping_add(slice.len())
            } else {
                // SAFETY: Caller ensures that this is valid.
                ptr.add(slice.len() * element_size)
            };

            Self {
                ptr: NonNull::new_unchecked(ptr),
                end,
                element_size,
                _marker: PhantomData,
            }
        }
    }
}

impl<'a> Iterator for TypeErasedIterMut<'a> {
    type Item = TypeErasedValueMut<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.as_ptr() as *const u8 == self.end {
            // Reach end of iterator
            return None;
        }

        unsafe {
            if self.element_size == 0 {
                // SAFETY: Caller guarantees that `ptr` is always valid
                let value = TypeErasedValueMut::new_unchecked(self.ptr.as_ptr());
                // For ZST the number of elements in this iterator is the kept as the difference in bytes
                // between the `end` and `ptr` pointers. So everytime we read a ZST element we bring
                // the `end` pointer one step closer to the `ptr` pointer.
                self.end = self.end.wrapping_sub(1);
                return Some(value);
            }

            // SAFETY: Caller guarantees that `ptr` is always valid
            let value = TypeErasedValueMut::new_unchecked(self.ptr.as_ptr());
            let ptr = self.ptr.as_ptr().add(self.element_size);
            self.ptr = NonNull::new_unchecked(ptr);
            Some(value)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.len();
        (exact, Some(exact))
    }
}

impl<'a> ExactSizeIterator for TypeErasedIterMut<'a> {
    #[inline]
    fn len(&self) -> usize {
        if self.element_size == 0 {
            unsafe { self.end.offset_from(self.ptr.as_ptr()) as usize }
        } else {
            // The pointers are expressed in bytes, and as such the offset difference needs to be divided
            // by the size of each element so we are left with the number of elements (not the number of bytes).
            unsafe { self.end.offset_from(self.ptr.as_ptr()) as usize / self.element_size }
        }
    }
}

impl<'a> IntoIterator for &'a TypeErasedVec {
    type Item = TypeErasedValueRef<'a>;
    type IntoIter = TypeErasedIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut TypeErasedVec {
    type Item = TypeErasedValueMut<'a>;
    type IntoIter = TypeErasedIterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// This function is outside [TypeErasedVec] to minimize compile times. See the comment
/// above `TypeErasedVec::grow_amortized` for details. (The `A` parameter isn't
/// significant, because the number of different `A` types seen in practice is
/// much smaller than the number of `T` types.)
///
/// # Remarks
/// Adapted from [std::vec::Vec].
#[inline(never)]
fn finish_grow(
    new_layout: Option<Layout>,
    current_memory: Option<(NonNull<u8>, Layout)>,
) -> Result<NonNull<u8>, TryReserveError> {
    // Check for the error here to minimize the size of `TypeErasedVec::grow_*`.
    let new_layout = new_layout.ok_or_else(|| TryReserveError::CapacityOverflow)?;

    alloc_guard(new_layout.size())?;

    let memory = if let Some((ptr, old_layout)) = current_memory {
        debug_assert_eq!(old_layout.align(), new_layout.align());
        unsafe {
            // The allocator checks for alignment equality
            // Unstable: intrinsics::assume(old_layout.align() == new_layout.align());
            std::alloc::realloc(ptr.as_ptr(), old_layout, new_layout.size())
        }
    } else {
        debug_assert!(new_layout.size() != 0);
        unsafe {
            // SAFETY: The calling contexts ensure that ZST are not used to allocate memory.
            std::alloc::alloc(new_layout)
        }
    };

    NonNull::new(memory).ok_or_else(|| TryReserveError::AllocError { layout: new_layout })
}

/// We need to guarantee the following:
/// * We don't ever allocate `> isize::MAX` byte-size objects.
/// * We don't overflow `usize::MAX` and actually allocate too little.
///
/// On 64-bit we just need to check for overflow since trying to allocate
/// `> isize::MAX` bytes will surely fail. On 32-bit and 16-bit we need to add
/// an extra guard for this in case we're running on a platform which can use
/// all 4GB in user-space, e.g., PAE or x32.
///
/// # Remarks
/// Adapted from [std::vec::Vec].
#[inline]
fn alloc_guard(alloc_size: usize) -> Result<(), TryReserveError> {
    if usize::BITS < 64 && alloc_size > isize::MAX as usize {
        Err(TryReserveError::CapacityOverflow)
    } else {
        Ok(())
    }
}

/// Error that occurrs when attempting to reserve memory in a collection.
///
/// # Remarks
/// Adapted from [std::vec::Vec].
pub enum TryReserveError {
    /// Error due to the computed capacity exceeding the collection's maximum
    /// (usually `isize::MAX` bytes).
    CapacityOverflow,

    /// The memory allocator returned an error
    AllocError {
        /// The layout of allocation request that failed
        layout: Layout,
    },
}

// Central function for reserve error handling.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn handle_reserve(result: Result<(), TryReserveError>) {
    use TryReserveError::{AllocError, CapacityOverflow};

    match result {
        Err(CapacityOverflow) => capacity_overflow(),
        Err(AllocError { layout }) => std::alloc::handle_alloc_error(layout),
        Ok(()) => {}
    }
}

// One central function responsible for reporting capacity overflows. This'll
// ensure that the code generation related to these panics is minimal as there's
// only one location which panics rather than a bunch throughout the module.
#[cfg(not(no_global_oom_handling))]
fn capacity_overflow() -> ! {
    panic!("capacity overflow");
}

#[cfg(test)]
mod tests {
    use std::alloc::Layout;

    use crate::{TypeErasedVec, TypedValue};

    #[test]
    fn it_creates_an_empty_vector_with_type() {
        let vec = TypeErasedVec::new::<i32>();
        assert_eq!(0, vec.len());
        assert_eq!(0, vec.capacity());
    }

    #[test]
    fn it_creates_an_empty_vector_with_layout() {
        let element_layout = Layout::new::<i32>();
        let vec = TypeErasedVec::empty(element_layout, None);
        assert_eq!(0, vec.len());
        assert_eq!(0, vec.capacity());
    }

    #[test]
    fn it_creates_an_empty_vector_with_capacity() {
        const CAP: usize = 10;

        let element_layout = Layout::new::<i32>();
        let vec = TypeErasedVec::with_capacity(element_layout, None, CAP);
        assert_eq!(0, vec.len());
        assert_eq!(CAP, vec.capacity());
    }

    #[test]
    fn it_creates_an_empty_vector_with_zst() {
        let vec = TypeErasedVec::new::<()>();
        assert_eq!(0, vec.len());
        assert_eq!(usize::MAX, vec.capacity());
    }

    #[test]
    fn it_reserves_space_in_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert!(vec.is_empty());

        vec.reserve(10);
        assert!(vec.capacity() >= 10);
    }

    #[test]
    fn it_reserves_space_in_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4, 5];
        assert!(vec.capacity() > 0);

        vec.reserve(10);
        assert!(vec.capacity() >= 10 + vec.len());
    }

    #[test]
    fn it_reserves_exact_space_in_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert!(vec.is_empty());

        vec.reserve_exact(10);
        assert!(vec.capacity() == 10);
    }

    #[test]
    fn it_reserves_exact_space_in_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4, 5];
        assert!(vec.capacity() > 0);

        vec.reserve_exact(10);
        assert!(vec.capacity() == 10 + vec.len());
    }

    #[test]
    fn it_pushes_value_into_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert_eq!(0, vec.len());
        assert_eq!(0, vec.capacity());

        unsafe {
            let value = TypedValue::new(42);
            vec.push(value);
        }

        assert_eq!(1, vec.len());
        assert!(vec.capacity() > 0);
    }

    #[test]
    fn it_pushes_value_into_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4, 5];
        assert_eq!(5, vec.len());

        unsafe {
            let value = TypedValue::new(42);
            vec.push(value);
        }

        assert_eq!(6, vec.len());
    }

    #[test]
    fn it_inserts_value_into_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert!(vec.is_empty());

        unsafe {
            vec.insert(0, TypedValue::new(1));

            assert_eq!(1, vec.len());
            assert_eq!(1, *vec.get_unchecked(0).get::<i32>());
        }
    }

    #[test]
    fn it_inserts_value_into_beginning_of_populated_vector() {
        let mut vec = type_erased_vec![i32, 2, 3, 4, 5];

        let len: usize = vec.len();
        unsafe {
            vec.insert(0, TypedValue::new(1));

            assert_eq!(len + 1, vec.len());
            assert_eq!(1, *vec.get_unchecked(0).get::<i32>());
        }
    }

    #[test]
    fn it_inserts_value_into_middle_of_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 4, 5];

        let len: usize = vec.len();
        unsafe {
            vec.insert(2, TypedValue::new(3));

            assert_eq!(len + 1, vec.len());
            assert_eq!(3, *vec.get_unchecked(2).get::<i32>());
        }
    }

    #[test]
    fn it_inserts_value_into_end_of_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4];

        let len: usize = vec.len();
        unsafe {
            vec.insert(4, TypedValue::new(5));

            assert_eq!(len + 1, vec.len());
            assert_eq!(5, *vec.get_unchecked(4).get::<i32>());
        }
    }

    #[test]
    #[should_panic]
    fn it_removes_value_from_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert!(vec.is_empty());
        vec.remove(0);
    }

    #[test]
    fn it_removes_value_from_beginning_of_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4, 5];

        let len: usize = vec.len();
        let value = vec.remove(0);
        assert_eq!(len - 1, vec.len());

        unsafe {
            assert_eq!(1, *value.get::<i32>());
            assert_eq!(2, *vec.get_unchecked(0).get::<i32>());
        }
    }

    #[test]
    fn it_removes_value_from_middle_of_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4, 5];

        let len: usize = vec.len();
        let value = vec.remove(2);
        assert_eq!(len - 1, vec.len());

        unsafe {
            assert_eq!(3, *value.get::<i32>());
            assert_eq!(4, *vec.get_unchecked(2).get::<i32>());
        }
    }

    #[test]
    fn it_removes_value_from_end_of_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4, 5];

        let len: usize = vec.len();
        let value = vec.remove(len - 1);
        assert_eq!(len - 1, vec.len());

        unsafe {
            assert_eq!(5, *value.get::<i32>());
            assert_eq!(4, *vec.get_unchecked(len - 2).get::<i32>());
        }
    }

    #[test]
    fn it_tries_to_remove_value_from_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert_eq!(0, vec.len());
        assert!(vec.try_remove(0).is_none());
    }

    #[test]
    fn it_tries_to_remove_value_from_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2];
        assert_eq!(2, vec.len());

        let value = vec.try_remove(1);
        assert_eq!(2, unsafe { *value.unwrap().get::<i32>() });

        let value = vec.try_remove(1);
        assert!(value.is_none());
    }

    #[test]
    #[should_panic]
    fn it_swap_removes_value_from_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert_eq!(0, vec.len());
        vec.swap_remove(0);
    }

    #[test]
    fn it_swap_removes_value_from_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3];
        assert_eq!(3, vec.len());

        let value = vec.swap_remove(0);
        assert_eq!(1, unsafe { *value.get::<i32>() });
        assert_eq!(2, vec.len());
    }

    #[test]
    #[should_panic]
    fn it_purges_value_from_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert_eq!(0, vec.len());

        vec.purge(0);
    }

    #[test]
    fn it_purges_value_from_beginning_of_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4, 5];
        assert!(!vec.is_empty());

        vec.purge(0);
        assert_eq!(4, vec.len());
        assert_eq!(2, unsafe { *vec.get_unchecked(0).get::<i32>() });
    }

    #[test]
    fn it_purges_value_from_middle_of_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4, 5];
        assert!(!vec.is_empty());

        vec.purge(2);
        assert_eq!(4, vec.len());
        assert_eq!(4, unsafe { *vec.get_unchecked(2).get::<i32>() });
    }

    #[test]
    fn it_purges_value_from_end_of_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4, 5];
        assert!(!vec.is_empty());

        vec.purge(4);
        assert_eq!(4, vec.len());
        assert_eq!(4, unsafe { *vec.get_unchecked(3).get::<i32>() });
    }

    #[test]
    fn it_tries_to_purge_value_from_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert_eq!(0, vec.len());
        assert!(!vec.try_purge(0));
    }

    #[test]
    fn it_tries_to_purge_value_from_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2];
        assert_eq!(2, vec.len());

        assert!(vec.try_purge(1));
        assert_eq!(1, vec.len());

        assert!(!vec.try_purge(1));
        assert_eq!(1, vec.len());
    }

    #[test]
    #[should_panic]
    fn it_swap_purges_value_from_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert_eq!(0, vec.len());
        vec.swap_purge(0);
    }

    #[test]
    fn it_swap_purges_value_from_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3];
        assert_eq!(3, vec.len());

        vec.swap_purge(0);
        assert_eq!(2, vec.len());
    }

    #[test]
    fn it_pops_value_from_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert_eq!(0, vec.len());

        let value = vec.pop();
        assert!(value.is_none());
    }

    #[test]
    fn it_pops_value_from_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4, 5];
        assert_eq!(5, vec.len());

        let value = vec.pop();
        assert!(value.is_some());

        let value = value.unwrap();
        let value = unsafe { value.get::<i32>() };
        assert_eq!(5, *value);
    }

    #[test]
    fn it_reads_value_from_empty_vector() {
        let vec = type_erased_vec![i32];
        assert!(vec.is_empty());

        let value = vec.get(0);
        assert!(value.is_none());
    }

    #[test]
    fn it_reads_value_from_populated_vector() {
        let vec = type_erased_vec![i32, 1, 2, 3, 4];
        assert!(!vec.is_empty());

        let value = vec.get(2);
        assert!(value.is_some());

        unsafe {
            assert_eq!(3, *value.unwrap().get::<i32>());
        }
    }

    #[test]
    fn it_reads_unchecked_value_from_populated_vector() {
        let vec = type_erased_vec![i32, 1, 2, 3, 4];
        assert!(!vec.is_empty());

        unsafe {
            let value = vec.get_unchecked(2);
            assert_eq!(3, *value.get::<i32>());
        }
    }

    #[test]
    fn it_reads_mutable_value_from_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert!(vec.is_empty());

        let value = vec.get_mut(0);
        assert!(value.is_none());
    }

    #[test]
    fn it_reads_mutable_value_from_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4];
        assert!(!vec.is_empty());

        let value = vec.get_mut(2);
        assert!(value.is_some());

        unsafe {
            assert_eq!(3, *value.unwrap().get_mut::<i32>());
        }
    }

    #[test]
    fn it_reads_unchecked_mutable_value_from_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4];
        assert!(!vec.is_empty());

        unsafe {
            let value = vec.get_unchecked_mut(2);
            assert_eq!(3, *value.get_mut::<i32>());
        }
    }

    #[test]
    fn it_clears_empty_vector() {
        let mut vec = type_erased_vec![i32];
        assert!(vec.is_empty());

        vec.clear();
        assert!(vec.is_empty());
    }

    #[test]
    fn it_clears_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4];
        assert!(!vec.is_empty());

        vec.clear();
        assert!(vec.is_empty());
    }

    #[test]
    fn it_iterates_empty_vector() {
        let vec = type_erased_vec![i32];

        let mut count = 0;
        for _ in &vec {
            count += 1;
        }

        assert_eq!(0, count);
    }

    #[test]
    fn it_iterates_populated_vector() {
        let vec = type_erased_vec![i32, 1, 2, 3, 4, 5];

        let mut count = 0;
        for _ in &vec {
            count += 1;
        }

        assert_eq!(5, count);
    }

    #[test]
    fn it_mut_iterates_empty_vector() {
        let mut vec = type_erased_vec![i32];

        let mut count = 0;
        for _ in &mut vec {
            count += 1;
        }

        assert_eq!(0, count);
    }

    #[test]
    fn it_mut_iterates_populated_vector() {
        let mut vec = type_erased_vec![i32, 1, 2, 3, 4, 5];

        let mut count = 0;
        for _ in &mut vec {
            count += 1;
        }

        assert_eq!(5, count);
    }
}
