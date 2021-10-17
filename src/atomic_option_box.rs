use atomic_box_base::{AtomicBoxBase, Handle, HandleReferable, PointerConvertible};
use std::fmt::{self, Debug, Formatter};
use std::ptr::null_mut;
use std::sync::atomic::Ordering;

impl<T> PointerConvertible for Option<Box<T>> {
    type Target = T;

    fn into_raw(b: Self) -> *mut T {
        match b {
            Some(box_value) => Box::into_raw(box_value),
            None => null_mut(),
        }
    }

    unsafe fn from_raw(ptr: *mut T) -> Self {
        if ptr.is_null() {
            None
        } else {
            Some(Box::from_raw(ptr))
        }
    }
}

impl<T> HandleReferable for Option<Box<T>> {
    type Target = T;

    fn make_handle(&self) -> Handle<T> {
        match self {
            Some(box_value) => box_value.make_handle(),
            None => Handle { ptr: null_mut() },
        }
    }
}

/// A type that holds a single `Option<Box<T>>` value and can be safely shared
/// between threads.
pub struct AtomicOptionBox<T> {
    base: AtomicBoxBase<Option<Box<T>>>,
}

impl<T> AtomicOptionBox<T> {
    /// Creates a new `AtomicOptionBox` with the given value.
    ///
    /// # Examples
    ///
    ///     use atomicbox::AtomicOptionBox;
    ///
    ///     let atomic_box = AtomicOptionBox::new(Some(Box::new(0)));
    ///
    pub fn new(value: Option<Box<T>>) -> AtomicOptionBox<T> {
        AtomicOptionBox {
            base: AtomicBoxBase::new(value),
        }
    }

    /// Atomically set this `AtomicOptionBox` to `other` and return the
    /// previous value.
    ///
    /// This does not allocate or free memory, and it neither clones nor drops
    /// any values.  `other` is moved into `self`; the value previously in
    /// `self` is returned.
    ///
    /// `ordering` must be either `Ordering::AcqRel` or `Ordering::SeqCst`,
    /// as other values would not be safe if `T` contains any data.
    ///
    /// # Panics
    ///
    /// Panics if `ordering` is not one of the two allowed values.
    ///
    /// # Examples
    ///
    ///     use std::sync::atomic::Ordering;
    ///     use atomicbox::AtomicOptionBox;
    ///
    ///     let atom = AtomicOptionBox::new(None);
    ///     let prev_value = atom.swap(Some(Box::new("ok")), Ordering::AcqRel);
    ///     assert_eq!(prev_value, None);
    ///
    pub fn swap(&self, other: Option<Box<T>>, order: Ordering) -> Option<Box<T>> {
        self.base.swap(other, order)
    }

    /// Atomically set this `AtomicOptionBox` to `other` and drop the
    /// previous value.
    ///
    /// The `AtomicOptionBox` takes ownership of `other`.
    ///
    /// `ordering` must be either `Ordering::AcqRel` or `Ordering::SeqCst`,
    /// as other values would not be safe if `T` contains any data.
    ///
    /// # Panics
    ///
    /// Panics if `ordering` is not one of the two allowed values.
    ///
    /// # Examples
    ///
    ///     use std::sync::atomic::Ordering;
    ///     use atomicbox::AtomicOptionBox;
    ///
    ///     let atom = AtomicOptionBox::new(None);
    ///     atom.store(Some(Box::new("ok")), Ordering::AcqRel);
    ///     assert_eq!(atom.into_inner(), Some(Box::new("ok")));
    ///
    pub fn store(&self, other: Option<Box<T>>, order: Ordering) {
        self.base.store(other, order)
    }

    /// Atomically set this `AtomicOptionBox` to `None` and return the
    /// previous value.
    ///
    /// This does not allocate or free memory, and it neither clones nor drops
    /// any values. It is equivalent to calling `self.swap(None, order)`
    ///
    /// `ordering` must be either `Ordering::AcqRel` or `Ordering::SeqCst`,
    /// as other values would not be safe if `T` contains any data.
    ///
    /// # Panics
    ///
    /// Panics if `ordering` is not one of the two allowed values.
    ///
    /// # Examples
    ///
    ///     use std::sync::atomic::Ordering;
    ///     use atomicbox::AtomicOptionBox;
    ///
    ///     let atom = AtomicOptionBox::new(Some(Box::new("ok")));
    ///     let prev_value = atom.take(Ordering::AcqRel);
    ///     assert!(prev_value.is_some());
    ///     let prev_value = atom.take(Ordering::AcqRel);
    ///     assert!(prev_value.is_none());
    ///
    pub fn take(&self, order: Ordering) -> Option<Box<T>> {
        self.base.swap(None, order)
    }

    /// Atomically swaps the contents of this `AtomicOptionBox` with the contents of `other`.
    ///
    /// This does not allocate or free memory, and it neither clones nor drops
    /// any values. The pointers in `*other` and `self` are simply exchanged.
    ///
    /// `ordering` must be either `Ordering::AcqRel` or `Ordering::SeqCst`,
    /// as other values would not be safe if `T` contains any data.
    ///
    /// # Panics
    ///
    /// Panics if `ordering` is not one of the two allowed values.
    ///
    /// # Examples
    ///
    ///     use std::sync::atomic::Ordering;
    ///     use atomicbox::AtomicOptionBox;
    ///
    ///     let atom = AtomicOptionBox::new(None);
    ///     let mut boxed = Some(Box::new("ok"));
    ///     let prev_value = atom.swap_mut(&mut boxed, Ordering::AcqRel);
    ///     assert_eq!(boxed, None);
    ///
    pub fn swap_mut(&self, other: &mut Option<Box<T>>, order: Ordering) {
        self.base.swap_mut(other, order)
    }

    /// Consume this `AtomicOptionBox`, returning the last option value it
    /// contained.
    ///
    /// # Examples
    ///
    ///     use atomicbox::AtomicOptionBox;
    ///
    ///     let atom = AtomicOptionBox::new(Some(Box::new("hello")));
    ///     assert_eq!(atom.into_inner(), Some(Box::new("hello")));
    ///
    pub fn into_inner(self) -> Option<Box<T>> {
        self.base.into_inner()
    }

    /// Returns a mutable reference to the contained value.
    ///
    /// This is safe because it borrows the `AtomicOptionBox` mutably, which
    /// ensures that no other threads can concurrently access either the atomic
    /// pointer field or the boxed data it points to.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        // I have a convoluted theory that Relaxed is good enough here.
        // See comment in AtomicBox::get_mut().
        let ptr = self.base.ptr.load(Ordering::Relaxed);
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &mut *(ptr as *mut T) })
        }
    }

    /// Returns a handle that matches the current held option.
    /// load takes an Ordering argument which describes the memory ordering of this operation. Possible values are SeqCst, Acquire and Relaxed.
    pub fn load_handle(&self, order: Ordering) -> Handle<T> {
        self.base.load_handle(order)
    }

    /// Returns a pointer to the currently held box. Using the pointer is unsafe for all of the usual reasons; e.g., the box might have been dropped from
    /// the atomic by another thread by the time the pointer is dereferenced. If the current option is None, the returned pointer will be a null pointer.
    /// load takes an Ordering argument which describes the memory ordering of this operation. Possible values are SeqCst, Acquire and Relaxed.
    ///
    /// # Examples
    ///
    ///     use atomicbox::AtomicOptionBox;
    ///     use std::sync::atomic::Ordering;
    ///
    ///     let mut box1 = Box::new("goodbye");
    ///     let atom = AtomicOptionBox::new(None);
    ///     let ptr = &mut *box1 as *mut &str;
    ///
    ///     atom.store(Some(box1), Ordering::SeqCst);
    ///     assert_eq!(atom.load_pointer(Ordering::Relaxed), ptr);
    ///
    pub fn load_pointer(&self, order: Ordering) -> *mut T {
        self.base.load_pointer(order)
    }

    /// Stores a box into the atomic if the value held by the atomic matches the given current handle.
    /// The return value is a result indicating whether the new box was written and containing the previous box.
    /// On success the returned handle is guaranteed to match the current value.
    /// On failure, the returned value contains the handle that matches the value in the atomic, and the given new box.
    /// compare_exchange takes two Ordering arguments to describe the memory ordering of this operation. success describes the required ordering for the read-modify-write operation that takes place if the comparison with current succeeds. failure describes the required ordering for the load operation that takes place when the comparison fails. Using Acquire as success ordering makes the store part of this operation Relaxed, and using Release makes the successful load Relaxed. The failure ordering can only be SeqCst, Acquire or Relaxed and must be equivalent to or weaker than the success ordering.
    /// Note: This method is only available on platforms that support atomic operations on pointers.
    ///
    /// # Examples
    ///
    ///     use atomicbox::AtomicOptionBox;
    ///     use std::sync::atomic::Ordering;
    ///
    ///     let atom = AtomicOptionBox::new(None);
    ///     let mut box1 = Some(Box::new("goodbye"));
    ///     let current = atom.load_handle(Ordering::Relaxed);
    ///     let result = atom.compare_exchange(current, box1, Ordering::SeqCst, Ordering::Relaxed);
    ///     assert_eq!(result, Ok(None));
    ///
    pub fn compare_exchange(
        &self,
        current: Handle<T>,
        new: Option<Box<T>>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Option<Box<T>>, (Handle<T>, Option<Box<T>>)> {
        self.base.compare_exchange(current, new, success, failure)
    }

    /// Stores an optional box into the atomic and moves the value in the new box into the atomic, if the value held by the atomic matches the given current handle.
    /// The return value is a result indicating whether the new box was written.
    /// On success the returned handle is guaranteed to match the current value.
    /// On failure, the returned value contains the handle that matches the value in the atomic, and new isn't mutated.
    /// compare_exchange_mut takes two Ordering arguments to describe the memory ordering of this operation. success describes the required ordering for the read-modify-write operation that takes place if the comparison with current succeeds. failure describes the required ordering for the load operation that takes place when the comparison fails. Using Acquire as success ordering makes the store part of this operation Relaxed, and using Release makes the successful load Relaxed. The failure ordering can only be SeqCst, Acquire or Relaxed and must be equivalent to or weaker than the success ordering.
    /// Note: This method is only available on platforms that support atomic operations on pointers.
    ///
    /// # Examples
    ///
    ///     use atomicbox::AtomicOptionBox;
    ///     use std::sync::atomic::Ordering;
    ///
    ///     let atom = AtomicOptionBox::new(None);
    ///     let mut box1 = Some(Box::new("goodbye"));
    ///     let current = atom.load_handle(Ordering::Relaxed);
    ///     atom.compare_exchange_mut(current, &mut box1, Ordering::SeqCst, Ordering::Relaxed);
    ///     assert_eq!(box1, None);
    ///
    pub fn compare_exchange_mut(
        &self,
        current: Handle<T>,
        new: &mut Option<Box<T>>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Handle<T>, Handle<T>> {
        self.base
            .compare_exchange_mut(current, new, success, failure)
    }

    /// Stores an optional box into the atomic if the value held by the atomic matches the given current handle.
    /// Unlike AtomicBox::compare_exchange, this function is allowed to spuriously fail even when the comparison succeeds, which can result in more efficient code on some platforms.
    /// The return value is a result indicating whether the new box was written and containing the previous box.
    /// On success the returned handle is guaranteed to match the current value.
    /// On failure, the returned value contains the handle that matches the value in the atomic and the given new box.
    /// compare_exchange_weak takes two Ordering arguments to describe the memory ordering of this operation. success describes the required ordering for the read-modify-write operation that takes place if the comparison with current succeeds. failure describes the required ordering for the load operation that takes place when the comparison fails. Using Acquire as success ordering makes the store part of this operation Relaxed, and using Release makes the successful load Relaxed. The failure ordering can only be SeqCst, Acquire or Relaxed and must be equivalent to or weaker than the success ordering.
    /// Note: This method is only available on platforms that support atomic operations on pointers.
    ///
    /// # Examples
    ///
    ///     use atomicbox::AtomicOptionBox;
    ///     use std::sync::atomic::Ordering;
    ///
    ///     let atom = AtomicOptionBox::new(None);
    ///     let mut box1 = Some(Box::new("goodbye"));
    ///     let mut current = atom.load_handle(Ordering::Relaxed);
    ///     box1 = loop {
    ///         match atom.compare_exchange_weak(current, box1, Ordering::SeqCst, Ordering::Relaxed) {
    ///             Ok(b) => break b,
    ///             Err((c, b)) => {
    ///                 current = c;
    ///                 box1 = b;
    ///             }
    ///         }
    ///     };
    ///
    pub fn compare_exchange_weak(
        &self,
        current: Handle<T>,
        new: Option<Box<T>>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Option<Box<T>>, (Handle<T>, Option<Box<T>>)> {
        self.base
            .compare_exchange_weak(current, new, success, failure)
    }

    /// Stores an optional box into the atomic and moves the value in the new box into the atomic, if the value held by the atomic matches the given current handle.
    /// Unlike AtomicBox::compare_exchange_mut, this function is allowed to spuriously fail even when the comparison succeeds, which can result in more efficient code on some platforms.
    /// The return value is a result indicating whether the new optional was written.
    /// On success the returned handle is guaranteed to match the current value.
    /// On failure, the returned value contains the handle that matches the value in the atomic, and new isn't mutated.
    /// compare_exchange_weak_mut takes two Ordering arguments to describe the memory ordering of this operation. success describes the required ordering for the read-modify-write operation that takes place if the comparison with current succeeds. failure describes the required ordering for the load operation that takes place when the comparison fails. Using Acquire as success ordering makes the store part of this operation Relaxed, and using Release makes the successful load Relaxed. The failure ordering can only be SeqCst, Acquire or Relaxed and must be equivalent to or weaker than the success ordering.
    /// Note: This method is only available on platforms that support atomic operations on pointers.
    ///
    /// # Examples
    ///
    ///     use atomicbox::AtomicOptionBox;
    ///     use std::sync::atomic::Ordering;
    ///
    ///     let atom = AtomicOptionBox::new(None);
    ///     let mut box1 = Some(Box::new("hello"));
    ///     let current = atom.load_handle(Ordering::Relaxed);
    ///     loop {
    ///         let result = atom.compare_exchange_weak_mut(current, &mut box1, Ordering::SeqCst, Ordering::Relaxed);
    ///         if result.is_ok() {
    ///             break;
    ///         }
    ///     }
    ///
    pub fn compare_exchange_weak_mut(
        &self,
        current: Handle<T>,
        new: &mut Option<Box<T>>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Handle<T>, Handle<T>> {
        self.base
            .compare_exchange_weak_mut(current, new, success, failure)
    }
}

impl<T> Default for AtomicOptionBox<T> {
    /// The default `AtomicOptionBox<T>` value is `AtomicBox::new(None)`.
    fn default() -> AtomicOptionBox<T> {
        AtomicOptionBox::new(None)
    }
}

impl<T> Debug for AtomicOptionBox<T> {
    /// The `{:?}` format of an `AtomicOptionBox<T>` looks like
    /// `"AtomicOptionBox(0x12341234)"` or `"AtomicOptionBox(None)"`.
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        let p = self.base.ptr.load(Ordering::Relaxed);
        f.write_str("AtomicOptionBox(")?;
        if p.is_null() {
            f.write_str("None")?;
        } else {
            fmt::Pointer::fmt(&p, f)?;
        }
        f.write_str(")")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Barrier};
    use std::thread::spawn;

    #[test]
    fn atomic_option_box_swap_works() {
        let b = AtomicOptionBox::new(Some(Box::new("hello world")));
        let bis = Box::new("bis");
        assert_eq!(
            b.swap(None, Ordering::AcqRel),
            Some(Box::new("hello world"))
        );
        assert_eq!(b.swap(Some(bis), Ordering::AcqRel), None);
        assert_eq!(b.swap(None, Ordering::AcqRel), Some(Box::new("bis")));
    }

    #[test]
    fn atomic_option_box_store_works() {
        let b = AtomicOptionBox::new(Some(Box::new("hello world")));
        b.store(None, Ordering::AcqRel);
        assert_eq!(b.into_inner(), None);

        let b = AtomicOptionBox::new(Some(Box::new("hello world")));
        let bis = Box::new("bis");
        b.store(Some(bis), Ordering::AcqRel);
        assert_eq!(b.into_inner(), Some(Box::new("bis")));
    }

    #[test]
    fn atomic_option_box_swap_mut_works() {
        let b = AtomicOptionBox::new(Some(Box::new("hello world")));
        let mut bis = None;
        b.swap_mut(&mut bis, Ordering::AcqRel);
        assert_eq!(bis, Some(Box::new("hello world")));
        bis = Some(Box::new("bis"));
        b.swap_mut(&mut bis, Ordering::AcqRel);
        assert_eq!(bis, None);
        b.swap_mut(&mut bis, Ordering::AcqRel);
        assert_eq!(bis, Some(Box::new("bis")));
    }

    #[test]
    fn atomic_option_box_pointer_identity() {
        let box1 = Box::new(1);
        let p1 = format!("{:p}", box1);
        let atom = AtomicOptionBox::new(Some(box1));

        let box2 = Box::new(2);
        let p2 = format!("{:p}", box2);
        assert!(p2 != p1);

        let box3 = atom.swap(Some(box2), Ordering::AcqRel).unwrap(); // box1 out, box2 in
        let p3 = format!("{:p}", box3);
        assert_eq!(p3, p1); // box3 is box1

        let box4 = atom.swap(None, Ordering::AcqRel).unwrap(); // box2 out, None in
        let p4 = format!("{:p}", box4);
        assert_eq!(p4, p2); // box4 is box2
    }

    #[test]
    fn atomic_box_drops() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct K(Arc<AtomicUsize>, usize);

        impl Drop for K {
            fn drop(&mut self) {
                self.0.fetch_add(self.1, Ordering::Relaxed);
            }
        }

        let n = Arc::new(AtomicUsize::new(0));
        {
            let ab = AtomicOptionBox::new(Some(Box::new(K(n.clone(), 5))));
            assert_eq!(n.load(Ordering::Relaxed), 0);
            let first = ab.swap(None, Ordering::AcqRel);
            assert_eq!(n.load(Ordering::Relaxed), 0);
            drop(first);
            assert_eq!(n.load(Ordering::Relaxed), 5);
            let second = ab.swap(Some(Box::new(K(n.clone(), 13))), Ordering::AcqRel);
            assert_eq!(second.is_none(), true);
            assert_eq!(n.load(Ordering::Relaxed), 5);
        }
        assert_eq!(n.load(Ordering::Relaxed), 5 + 13);
    }

    #[test]
    fn lockless_list() {
        use std::mem::forget;

        struct Node(AtomicOptionBox<Node>);

        const NTHREADS: usize = 9;
        const NITERATIONS: usize = 100;

        let gate = Arc::new(Barrier::new(NTHREADS));
        let mut head: Arc<AtomicOptionBox<Node>> = Arc::new(Default::default());
        let handles: Vec<_> = (0..NTHREADS as u8)
            .map(|_| {
                let my_gate = gate.clone();
                let my_head = head.clone();
                spawn(move || {
                    my_gate.wait();
                    for _ in 0..NITERATIONS {
                        let mut node_box = Box::new(Node(AtomicOptionBox::new(None)));
                        let mut current_head_handle = my_head.load_handle(Ordering::Acquire);
                        loop {
                            let current_head_box =
                                unsafe { Box::from_raw(current_head_handle.ptr as *mut Node) };
                            let prev_head_box =
                                node_box.0.swap(Some(current_head_box), Ordering::AcqRel);
                            forget(prev_head_box);
                            let result = my_head.compare_exchange_weak(
                                current_head_handle,
                                Some(node_box),
                                Ordering::SeqCst,
                                Ordering::Relaxed,
                            );
                            match result {
                                Ok(old_box) => {
                                    forget(old_box);
                                    break;
                                }
                                Err((previous_handle, sent_box)) => {
                                    current_head_handle = previous_handle;
                                    node_box = sent_box.unwrap()
                                }
                            }
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap()
        }
        let mut count: usize = 0;
        let mut node = Arc::get_mut(&mut head).unwrap();
        while let Some(next_node) = node.get_mut() {
            node = &mut next_node.0;
            count += 1;
        }

        assert_eq!(count, NTHREADS * NITERATIONS);
    }

    #[test]
    #[should_panic(expected = "invalid ordering for atomic swap")]
    fn cant_use_foolish_swap_ordering_type() {
        let atom = AtomicOptionBox::new(Some(Box::new(0)));
        atom.swap(None, Ordering::Release); // nope
    }

    #[test]
    fn debug_fmt() {
        let my_box = Box::new(32);
        let expected = format!("AtomicOptionBox({:p})", my_box);
        assert_eq!(
            format!("{:?}", AtomicOptionBox::new(Some(my_box))),
            expected
        );
        assert_eq!(
            format!("{:?}", AtomicOptionBox::<String>::new(None)),
            "AtomicOptionBox(None)"
        );
    }
}
