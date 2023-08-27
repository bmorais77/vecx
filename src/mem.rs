use std::alloc::Layout;

/// Performs an optimized `copy_nonoverlapping` without knowing the underlying type.
///
/// # Safety
///
/// Same safety guarantees that [std::ptr::copy_nonoverlapping] makes.
#[inline]
pub unsafe fn copy_nonoverlapping(dst: *mut u8, src: *const u8, count: usize) {
    if count <= SMALL_COPY_THRESHOLD {
        small_copy(dst, src, count);
    } else {
        std::ptr::copy_nonoverlapping(src, dst, count);
    }
}

/// Performs an optimized `copy_nonoverlapping` with the underlying data type known.
#[inline]
pub unsafe fn typed_copy_nonoverlapping<T>(dst: *mut T, src: *const T) {
    if std::mem::size_of::<T>() <= SMALL_COPY_THRESHOLD {
        small_copy(dst as *mut u8, src as *const u8, std::mem::size_of::<T>());
    } else {
        std::ptr::copy_nonoverlapping(src, dst, 1);
    }
}

/// Number of bytes where a "small copy" is no longer preferred.
const SMALL_COPY_THRESHOLD: usize = 32;

/// Performs a non-overlapping copy using a slightly more optimized procedure.
/// Note: This function does nothing if `count > SMALL_COPY_THRESHOLD`.
#[inline]
unsafe fn small_copy(dst: *mut u8, src: *const u8, count: usize) {
    if count < 2 {
        *dst = *src;
        return;
    }

    if count <= 4 {
        let a = src.cast::<u16>().read_unaligned();
        let b = src.add(count - 2).cast::<u16>().read_unaligned();
        dst.cast::<u16>().write_unaligned(a);
        dst.add(count - 2).cast::<u16>().write_unaligned(b);
        return;
    }

    if count <= 8 {
        let a = src.cast::<u32>().read_unaligned();
        let b = src.add(count - 4).cast::<u32>().read_unaligned();
        dst.cast::<u32>().write_unaligned(a);
        dst.add(count - 4).cast::<u32>().write_unaligned(b);
        return;
    }

    if count <= 16 {
        let a = src.cast::<u64>().read_unaligned();
        let b = src.add(count - 8).cast::<u64>().read_unaligned();
        dst.cast::<u64>().write_unaligned(a);
        dst.add(count - 8).cast::<u64>().write_unaligned(b);
        return;
    }

    if count <= 32 {
        let a = src.cast::<u128>().read_unaligned();
        let b = src.add(count - 16).cast::<u128>().read_unaligned();
        dst.cast::<u128>().write_unaligned(a);
        dst.add(count - 16).cast::<u128>().write_unaligned(b);
        return;
    }
}

/// Creates a layout describing the record for a `[T; n]` where `T`'s memory layout is described by
/// the given `element_layout`.
///
/// On arithmetic overflow or when the total size would exceed
/// `isize::MAX`, returns `None`.
///
/// # Arguments
/// * `element_layout` - Memory layout of the elements in the array.
/// * `n` - Number of elements in the array.
#[inline]
pub fn array_layout(element_layout: Layout, n: usize) -> Option<Layout> {
    #[inline(always)]
    const fn max_size_for_align(align: usize) -> usize {
        // (power-of-two implies align != 0.)

        // Rounded up size is:
        //   size_rounded_up = (size + align - 1) & !(align - 1);
        //
        // We know from above that align != 0. If adding (align - 1)
        // does not overflow, then rounding up will be fine.
        //
        // Conversely, &-masking with !(align - 1) will subtract off
        // only low-order-bits. Thus if overflow occurs with the sum,
        // the &-mask cannot subtract enough to undo that overflow.
        //
        // Above implies that checking for summation overflow is both
        // necessary and sufficient.
        isize::MAX as usize - (align - 1)
    }

    let element_size = element_layout.size();
    let align = element_layout.align();

    // We need to check two things about the size:
    //  - That the total size won't overflow a `usize`, and
    //  - That the total size still fits in an `isize`.
    // By using division we can check them both with a single threshold.
    // That'd usually be a bad idea, but thankfully here the element size
    // and alignment are constants, so the compiler will fold all of it.
    if element_size != 0 && n > max_size_for_align(align) / element_size {
        return None;
    }

    let array_size = element_size * n;

    // SAFETY: We just checked above that the `array_size` will not
    // exceed `isize::MAX` even when rounded up to the alignment.
    // And `Alignment` guarantees it's a power of two.
    unsafe { Some(Layout::from_size_align_unchecked(array_size, align)) }
}
