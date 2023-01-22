#![feature(test)]

extern crate test;

use matrix::Matrix;

/// Naively multiplies matrix `a` by matrix `b`, producing `c = a * b`.
///
/// # Panics
///
/// Panics if `a.num_columns() != b.num_rows()`.
pub fn matmul_naive(a: &Matrix<f64>, b: &Matrix<f64>) -> Matrix<f64> {
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

    let mut c: Matrix<f64> = Matrix::zeros(a.num_rows(), b.num_columns());

    for i in 0..a.num_rows() {
        let ai = &a[i];
        let ci = &mut c[i];

        for j in 0..b.num_columns() {
            let mut res = 0.;

            for k in 0..a.num_columns() {
                res += ai[k] * b[k][j];
            }

            ci[j] = res;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::Matrix;
    use test::Bencher;

    const NUM_ROWS: usize = 1_000;
    const NUM_COLUMNS: usize = 1_000;

    #[test]
    fn naive_matrix_multiplication() {
        let a: Matrix<f64> = Matrix::from([[0., 1.], [2., 3.], [4., 5.]]);
        let b: Matrix<f64> = Matrix::from([[6.], [7.]]);
        let c = matmul_naive(&a, &b);

        assert_eq!(c.num_rows(), a.num_rows());
        assert_eq!(c.num_columns(), b.num_columns());
        assert_eq!(c, Matrix::from([[7.], [33.], [59.]]));
    }

    #[bench]
    fn matmul_benchmark(b: &mut Bencher) {
        let a: Matrix<f64> = Matrix::ones(NUM_ROWS, NUM_COLUMNS);

        b.iter(|| Matrix::mul(&a, &a));
    }

    #[bench]
    fn matmul_naive_benchmark(b: &mut Bencher) {
        let a: Matrix<f64> = Matrix::ones(NUM_ROWS, NUM_COLUMNS);

        b.iter(|| matmul_naive(&a, &a));
    }
}
