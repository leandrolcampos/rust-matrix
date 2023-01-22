use super::Matrix;
use num_traits::Zero;
use rayon::prelude::*;
use std::ops::{AddAssign, Mul};

impl<T> Matrix<T> {
    /// Multiplies matrix `a` by matrix `b`, producing `c = a * b`.
    ///
    /// # Panics
    ///
    /// Panics if `a.num_columns() != b.num_rows()`.
    pub fn mul(a: &Self, b: &Self) -> Self
    where
        T: Copy + AddAssign + Mul<Output = T> + Zero + Sync + Send,
    {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(a_num_columns: usize, b_num_rows: usize) -> ! {
            panic!(
                "`a.num_columns()` (is {a_num_columns}) \
                should be equal to `b.num_rows()` (is {b_num_rows})"
            );
        }

        if a.num_columns() != b.num_rows() {
            assert_failed(a.num_columns(), b.num_rows());
        }

        let mut c: Matrix<T> = Matrix::zeros(a.num_rows(), b.num_columns());

        c.rows_mut()
            .zip(a.rows())
            .par_bridge()
            .for_each(|(ci, ai)| {
                b.rows().zip(ai.iter()).for_each(|(bk, aik)| {
                    ci.iter_mut().zip(bk.iter()).for_each(|(cij, bkj)| {
                        (*cij) += (*aik) * (*bkj);
                    })
                })
            });
        c
    }
}

#[cfg(test)]
mod test_mul {
    use super::Matrix;

    #[test]
    #[should_panic(expected = "`a.num_columns()` (is 2) \
                    should be equal to `b.num_rows()` (is 3)")]
    fn incompatible_shapes() {
        let a: Matrix<f32> = Matrix::from([[0., 1.], [2., 3.], [4., 5.]]);
        let b: Matrix<f32> = Matrix::from([[6.], [7.], [8.]]);
        let _c = Matrix::mul(&a, &b);
    }

    #[test]
    fn result() {
        let a: Matrix<f32> = Matrix::from([[0., 1.], [2., 3.], [4., 5.]]);
        let b: Matrix<f32> = Matrix::from([[6.], [7.]]);
        let c = Matrix::mul(&a, &b);

        assert_eq!(c.num_rows(), a.num_rows());
        assert_eq!(c.num_columns(), b.num_columns());
        assert_eq!(c, Matrix::from([[7.], [33.], [59.]]));
    }
}
