use std::{
    collections::HashMap,
    fmt::{self, Debug},
    hash::Hash,
    marker::PhantomData,
    ops,
    slice::Iter,
};

use bitvec::vec::BitVec;
use derivative::Derivative;

pub trait DenseKey {
    /// Don't use this.
    fn index(&self) -> usize;
}

#[derive(Derivative)]
#[derivative(PartialEq)]
#[derivative(Eq)]
#[derivative(Hash)]
#[derivative(Clone)]
#[derivative(Copy)]
#[derivative(Debug)]
pub struct Id<T> {
    index: usize,
    generic: PhantomData<T>,
}

impl<T> DenseKey for Id<T> {
    fn index(&self) -> usize {
        self.index
    }
}

impl<T> Id<T> {
    pub const INVALID: Self = Self {
        index: usize::MAX,
        generic: PhantomData,
    };
}

#[derive(Derivative)]
#[derivative(PartialEq)]
#[derivative(Eq)]
#[derivative(Hash)]
#[derivative(Clone)]
#[derivative(Copy)]
pub struct UniqueId<T> {
    index: usize,
    generic: PhantomData<T>,
}

impl<T> UniqueId<T> {
    fn from_id(id: Id<T>) -> Self {
        Self {
            index: id.index,
            generic: PhantomData,
        }
    }

    fn into_id(self) -> Id<T> {
        Id {
            index: self.index,
            generic: PhantomData,
        }
    }
}

impl<T> DenseKey for UniqueId<T> {
    fn index(&self) -> usize {
        self.index
    }
}

#[derive(Derivative)]
#[derivative(Default(bound = ""))]
pub struct DenseMap<T> {
    store: Vec<T>,
}

impl<T> DenseMap<T> {
    pub fn add(&mut self, value: T) -> Id<T> {
        let index = self.store.len();
        self.store.push(value);
        Id {
            index,
            generic: PhantomData,
        }
    }
}

impl<T> ops::Index<Id<T>> for DenseMap<T> {
    type Output = T;

    fn index(&self, id: Id<T>) -> &Self::Output {
        &self.store[id.index]
    }
}

impl<T> ops::IndexMut<Id<T>> for DenseMap<T> {
    fn index_mut(&mut self, id: Id<T>) -> &mut Self::Output {
        &mut self.store[id.index]
    }
}

#[derive(Derivative)]
#[derivative(Default(bound = "T: Eq + Hash + Clone"))]
pub struct DenseInternedMap<T: Eq + Hash + Clone> {
    store: DenseMap<T>,
    check: HashMap<T, Id<T>>,
}

impl<T: Eq + Hash + Clone> DenseInternedMap<T> {
    pub fn insert(&mut self, value: T) -> UniqueId<T> {
        UniqueId::from_id(if let Some(id) = self.check.get(&value) {
            *id
        } else {
            let id = self.store.add(value.clone());
            self.check.insert(value, id);
            id
        })
    }

    pub fn keys(&self) -> impl Iterator<Item = UniqueId<T>> + '_ {
        self.check.values().map(|id| UniqueId::from_id(*id))
    }
}

impl<T: Eq + Hash + Clone> ops::Index<UniqueId<T>> for DenseInternedMap<T> {
    type Output = T;

    fn index(&self, id: UniqueId<T>) -> &Self::Output {
        &self.store[id.into_id()]
    }
}

impl<T: Eq + Hash + Clone> ops::IndexMut<UniqueId<T>> for DenseInternedMap<T> {
    fn index_mut(&mut self, id: UniqueId<T>) -> &mut Self::Output {
        &mut self.store[id.into_id()]
    }
}

#[derive(Derivative)]
#[derivative(Default(bound = ""))]
pub struct DenseSet<K>
where
    K: DenseKey,
{
    valid: BitVec,
    generic: PhantomData<K>,
}

impl<K: DenseKey> DenseSet<K> {
    pub fn insert(&mut self, id: K) {
        if self.valid.len() < id.index() + 1 {
            self.valid.resize(id.index() + 1, false);
        }
        self.valid.set(id.index(), true);
    }

    pub fn contains(&self, id: K) -> bool {
        if id.index() < self.valid.len() {
            self.valid[id.index()]
        } else {
            false
        }
    }

    pub fn is_subset_of(&self, other: &DenseSet<K>) -> bool {
        for i in 0..self.valid.len() {
            if self.valid[i] && (other.valid.len() <= i || !other.valid[i]) {
                return false;
            }
        }
        true
    }

    pub fn clear(&mut self) {
        self.valid.fill(false);
    }
}

impl<K: DenseKey> Debug for DenseSet<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Binary::fmt(&self.valid, f)
    }
}

#[derive(Derivative)]
#[derivative(Default(bound = ""))]
pub struct DenseAssociationMap<Original, Assoc> {
    store: Vec<Assoc>,
    valid: DenseSet<Id<Original>>,
    generic: PhantomData<(Original, Assoc)>,
}

impl<Original, Assoc: Default> DenseAssociationMap<Original, Assoc> {
    pub fn associate(&mut self, id: Id<Original>, value: Assoc) {
        if id.index >= self.store.len() {
            self.store.resize_with(id.index + 1, Default::default);
        }
        self.store[id.index] = value;
        self.valid.insert(id);
    }

    pub fn retrieve(&self, id: Id<Original>) -> &Assoc {
        assert!(self.valid.contains(id), "index out of bounds");
        &self.store[id.index]
    }

    pub fn clear(&mut self) {
        self.valid.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = (Id<Original>, &Assoc)> {
        DenseAssociationMapIter {
            iter: self.store.iter(),
            valid: &self.valid,
            index: 0,
        }
    }
}

pub struct DenseAssociationMapIter<'a, Original, Assoc> {
    iter: Iter<'a, Assoc>,
    valid: &'a DenseSet<Id<Original>>,
    index: usize,
}

impl<'a, Original, Assoc> Iterator for DenseAssociationMapIter<'a, Original, Assoc> {
    type Item = (Id<Original>, &'a Assoc);

    fn next(&mut self) -> Option<Self::Item> {
        for value in self.iter.by_ref() {
            let id = Id {
                index: self.index,
                generic: PhantomData,
            };
            self.index += 1;
            if self.valid.contains(id) {
                return Some((id, value));
            }
        }
        None
    }
}
