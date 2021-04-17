use std::mem::MaybeUninit;

pub fn collect_to_array<T, I: Iterator<Item = T>, const N: usize>(mut iter: I) -> [T; N] {
    let mut dest: MaybeUninit<[T; N]> = MaybeUninit::uninit();
    let mut ptr = dest.as_mut_ptr() as *mut T;

    unsafe {
        for _ in 0..N {
            ptr.write(iter.next().unwrap());
            ptr = ptr.add(1);
        }

        dest.assume_init()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect() {
        let a0123 = collect_to_array(0..4);
        assert_eq!(a0123, [0, 1, 2, 3]);
    }
}
