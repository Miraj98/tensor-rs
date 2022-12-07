use crate::{dim::Dimension, DataElement, Tensor, TensorBase};

pub trait TensorConstructors<Dtype>
where
    Dtype: DataElement,
{
    type S: Dimension;
    fn ones(dim: Self::S) -> Self;
    fn zeros(dim: Self::S) -> Self;
    fn from_elem(dim: Self::S, elem: Dtype) -> Self;
}

impl<S, Dtype> TensorConstructors<Dtype> for Tensor<S, Dtype>
where
    S: Dimension,
    Dtype: DataElement,
{
    type S = S;

    fn ones(dim: S) -> Self {
        let a = vec![Dtype::one(); dim.count()];
        TensorBase::from_vec(a, dim)
    }

    fn zeros(dim: S) -> Self {
        let a = vec![Dtype::zero(); dim.count()];
        TensorBase::from_vec(a, dim)
    }

    fn from_elem(dim: S, elem: Dtype) -> Self {
        let a = vec![elem; dim.count()];
        TensorBase::from_vec(a, dim)
    }
}

pub fn tensor<A: IntoTensor>(a: A) -> A::Output {
    <A as IntoTensor>::into_tensor(a)
}

impl<const N: usize, A: DataElement> IntoTensor for [A; N] {
    type Dtype = A;
    type Output = Tensor<[usize; 1], A>;

    fn into_tensor(a: Self) -> Self::Output {
        TensorBase::from_vec(a.to_vec(), [N])
    }
}

impl<const M: usize, const N: usize, A: DataElement> IntoTensor for [[A; N]; M] {
    type Dtype = A;
    type Output = Tensor<[usize; 2], A>;

    fn into_tensor(a: Self) -> Self::Output {
        let mut v = Vec::with_capacity(M * N);
        for i in 0..M {
            for j in 0..N {
                v.push(a[i][j]);
            }
        }
        TensorBase::from_vec(v, [M, N])
    }
}

impl<const M: usize, const N: usize, const P: usize, A: DataElement> IntoTensor
    for [[[A; P]; N]; M]
{
    type Dtype = A;
    type Output = Tensor<[usize; 3], A>;

    fn into_tensor(a: Self) -> Self::Output {
        let mut v = Vec::with_capacity(M * N * P);
        for i in 0..M {
            for j in 0..N {
                for k in 0..P {
                    v.push(a[i][j][k]);
                }
            }
        }
        TensorBase::from_vec(v, [M, N, P])
    }
}

impl<const M: usize, const N: usize, const P: usize, const Q: usize, A: DataElement>
    IntoTensor for [[[[A; Q]; P]; N]; M]
{
    type Dtype = A;
    type Output = Tensor<[usize; 4], A>;

    fn into_tensor(a: Self) -> Self::Output {
        let mut v = Vec::with_capacity(M * N * P * Q);
        for i in 0..M {
            for j in 0..N {
                for k in 0..P {
                    for l in 0..Q {
                        v.push(a[i][j][k][l]);
                    }
                }
            }
        }
        TensorBase::from_vec(v, [M, N, P, Q])
    }
}

pub trait IntoTensor {
    type Dtype: DataElement;
    type Output;

    fn into_tensor(a: Self) -> Self::Output;
}
