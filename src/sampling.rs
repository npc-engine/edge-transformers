use onnxruntime::ndarray;
use onnxruntime::Axis;
use onnxruntime::tensor::ndarray_tensor::NdArrayTensor;
use distributions::WeightedIndex;
use prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use onnxruntime::ndarray::{Array, ArrayView, Axis, Ix2};
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

pub trait Sampler {
    fn sample(&self, logits: ArrayView<f32, Ix2>) -> usize;
}

pub struct TopKSampler {
    k: usize,
    temperature: f32,
}

pub struct RandomSampler {
    temperature: f32,
}

pub struct ArgmaxSampler {}

impl TopKSampler {
    pub fn new(k: usize, temperature: f32) -> Self {
        Self { k, temperature }
    }
}

impl RandomSampler {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl ArgmaxSampler {
    pub fn new() -> Self {
        Self {}
    }
}

impl Sampler for TopKSampler {
    fn sample(&self, logits: ArrayView<f32, Ix2>) -> usize {
        let top_elements: Vec<Vec<(usize, f32)>> = logits
            .axis_iter(Axis(0))
            .map(|row| {
                let mut h = BinaryHeap::new();
                for (id, item) in row.iter().enumerate() {
                    h.push(Reverse(Elem {
                        value: *item,
                        position: id,
                    }));
                    if h.len() > self.k {
                        h.pop();
                    }
                }
                h.into_iter()
                    .map(|rev| (rev.0.position, rev.0.value))
                    .collect()
            })
            .collect();
        let mut rng = thread_rng();
        let mut sampled_ids = Vec::new();
        for top_elements in top_elements {
            let mut weights = Vec::new();
            for (id, value) in top_elements {
                weights.push((id, (value / self.temperature).exp()));
            }
            let dist = WeightedIndex::new(weights.iter().map(|(_, w)| *w)).unwrap();
            sampled_ids.push(dist.sample(&mut rng));
        }
        sampled_ids[0]
    }
}

impl Sampler for RandomSampler {
    fn sample(&self, logits: ArrayView<f32, Ix2>) -> usize {
        let mut rng = thread_rng();
        let mut sampled_ids = Vec::new();
        for row in logits.axis_iter(Axis(0)) {
            let mut weights = Vec::new();
            for (id, value) in row.iter().enumerate() {
                weights.push((id, (value / self.temperature).exp()));
            }
            let dist = WeightedIndex::new(weights.iter().map(|(_, w)| *w)).unwrap();
            sampled_ids.push(dist.sample(&mut rng));
        }
        sampled_ids[0]
    }
}

impl Sampler for ArgmaxSampler {
    fn sample(&self, logits: ArrayView<f32, Ix2>) -> usize {
        let mut sampled_ids = Vec::new();
        for row in logits.axis_iter(Axis(0)) {
            let mut max_value = f32::MIN;
            let mut max_id = 0;
            for (id, value) in row.iter().enumerate() {
                if *value > max_value {
                    max_value = *value;
                    max_id = id;
                }
            }
            sampled_ids.push(max_id);
        }
        sampled_ids[0]
    }
}

#[derive(Copy, Clone)]
struct Elem {
    value: f32,
    position: usize,
}

impl PartialEq<Self> for Elem {
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&(other.value))
    }
}

impl Eq for Elem {}

impl PartialOrd<Self> for Elem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Ord for Elem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value
            .partial_cmp(&other.value)
            .unwrap_or(Ordering::Equal)
    }
}

pub fn select_k(
    array: Array<f32, Ix2>,
    k: usize,
    axis: Axis,
) -> Vec<Vec<(usize, f32)>> {
    let other_axis = if axis.index() == 0 { Axis(1) } else { Axis(0) };
    let top_elements: Vec<Vec<(usize, f32)>> = array
        .axis_iter(other_axis)
        .map(|row| {
            let mut h = BinaryHeap::new();
            for (id, item) in row.iter().enumerate() {
                h.push(Reverse(Elem {
                    value: *item,
                    position: id,
                }));
                if h.len() > k {
                    h.pop();
                }
            }
            h.into_iter()
                .map(|rev| (rev.0.position, rev.0.value))
                .collect()
        })
        .collect();
    top_elements
}

pub fn sample(
    array: ArrayView<f32, Ix2>,
    k: usize,
    temp: f32,
    axis: Axis,
) -> Vec<usize> {
    let softmax_array = (array.map(|x| x / temp)).softmax(Axis(1));
    let top_elems = select_k(softmax_array, k, axis);
    let mut rng = thread_rng();
    let top_elem_id: Vec<usize> = top_elems
        .iter()
        .map(|row| {
            WeightedIndex::new(row.iter().map(|x| x.1).collect::<Vec<f32>>())
                .unwrap()
                .sample(&mut rng)
        })
        .collect();
    top_elem_id
        .iter()
        .zip(top_elems)
        .map(|(top_elem_id, top_elems)| top_elems[*top_elem_id].0)
        .collect()
}
