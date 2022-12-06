#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub struct UniqueId(usize);

pub(crate) fn unique_id() -> UniqueId {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    UniqueId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

// impl UniqueId {
//     pub(crate) fn as_u64(&self) -> u64 {
//         self.0 as u64
//     }
// }

pub trait HasUniqueId {
    fn id(&self) -> &UniqueId;
}

pub(crate) mod internal {
    pub trait ResetId {
        fn reset_id(&mut self);
    }
}