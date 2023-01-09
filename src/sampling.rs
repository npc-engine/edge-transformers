use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use onnxruntime::ndarray::{Array, ArrayView, Axis, Ix2};
use onnxruntime::tensor::ndarray_tensor::NdArrayTensor;
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use rand::thread_rng;

pub trait Sampler {
    fn sample(&self, logits: ArrayView<f32, Ix2>) -> Vec<u32>;
}

pub struct TopKSampler {
    k: usize,
    temperature: f32,
}

pub struct RandomSampler {
    temperature: f32,
}

pub struct TopPSampler {
    p: f32,
    temperature: f32,
}

pub struct ContrastiveSampler {
    alpha: f32,
    temperature: f32,
}

pub struct ArgmaxSampler {}

impl TopKSampler {
    pub fn new(k: usize, temperature: f32) -> Self {
        Self {
            k,
            temperature: if temperature == 0.0 {
                1e-12
            } else {
                temperature
            },
        }
    }
}

impl RandomSampler {
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature: if temperature == 0.0 {
                1e-12
            } else {
                temperature
            },
        }
    }
}

impl ArgmaxSampler {
    pub fn new() -> Self {
        Self {}
    }
}

impl TopPSampler {
    pub fn new(p: f32, temperature: f32) -> Self {
        let temperature = if temperature == 0.0 {
            1e-12
        } else {
            temperature
        };
        Self { p, temperature }
    }
}

impl Sampler for TopKSampler {
    fn sample(&self, logits: ArrayView<f32, Ix2>) -> Vec<u32> {
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
            let dist = WeightedIndex::new(weights.iter().map(|(_, w)| *w));
            sampled_ids.push(match dist {
                Ok(dist) => dist.sample(&mut rng) as u32,
                Err(_) => weights[0].0 as u32,
            });
        }
        sampled_ids.into_iter().map(|id| id as u32).collect()
    }
}

impl Sampler for RandomSampler {
    fn sample(&self, logits: ArrayView<f32, Ix2>) -> Vec<u32> {
        let mut rng = thread_rng();
        let mut sampled_ids = Vec::new();
        for row in logits.axis_iter(Axis(0)) {
            let mut weights = Vec::new();
            for (id, value) in row.iter().enumerate() {
                weights.push((id, (value / self.temperature).exp()));
            }
            let dist = WeightedIndex::new(weights.iter().map(|(_, w)| *w));
            sampled_ids.push(match dist {
                Ok(dist) => dist.sample(&mut rng) as u32,
                Err(_) => weights[0].0 as u32,
            });
        }
        sampled_ids.into_iter().map(|id| id as u32).collect()
    }
}

impl Sampler for ArgmaxSampler {
    fn sample(&self, logits: ArrayView<f32, Ix2>) -> Vec<u32> {
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
        sampled_ids.into_iter().map(|id| id as u32).collect()
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

pub fn select_k(array: Array<f32, Ix2>, k: usize, axis: Axis) -> Vec<Vec<(usize, f32)>> {
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

pub fn sample(array: ArrayView<f32, Ix2>, k: usize, temp: f32, axis: Axis) -> Vec<usize> {
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
