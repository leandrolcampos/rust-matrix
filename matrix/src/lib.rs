//!
#![allow(dead_code)]

mod iter;
mod oper;

use iter::Rows;
use iter::RowsMut;
use num_traits::{One, Zero};
use std::ops::{Index, IndexMut};

/// A two-dimensional array type, written as `Matrix<T>`.
#[derive(Debug, PartialEq)]
pub struct Matrix<T> {
    data: Vec<T>,
    num_rows: usize,
    num_columns: usize,
}

impl<T> Matrix<T> {
    /// Creates a `Matrix<T>` with shape `(num_rows, num_columns)`, filled
    /// with `fill_value`.
    ///
    /// # Panics
    ///
    /// Panics if `num_rows` or `num_columns` equals zero.
    pub fn full(num_rows: usize, num_columns: usize, fill_value: T) -> Self
    where
        T: Copy,
    {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(dimension_name: &str) -> ! {
            panic!("`num_{dimension_name}` (is 0) should be > 0");
        }

        if num_rows == 0 {
            assert_failed("rows");
        }
        if num_columns == 0 {
            assert_failed("columns");
        }
        Self {
            data: vec![fill_value; num_rows * num_columns],
            num_rows,
            num_columns,
        }
    }

    /// Creates a `Matrix<T>` with shape `(num_rows, num_columns)`, filled
    /// with the default value of `T`.
    ///
    /// # Panics
    ///
    /// Panics if `num_rows` or `num_columns` equals zero.
    #[inline]
    pub fn new(num_rows: usize, num_columns: usize) -> Self
    where
        T: Copy + Default,
    {
        Self::full(num_rows, num_columns, T::default())
    }

    /// Creates a `Matrix<T>` with shape `(num_rows, num_columns)`, filled
    /// with zeros.
    ///
    /// # Panics
    ///
    /// Panics if `num_rows` or `num_columns` equals zero.
    #[inline]
    pub fn zeros(num_rows: usize, num_columns: usize) -> Self
    where
        T: Copy + Zero,
    {
        Self::full(num_rows, num_columns, T::zero())
    }

    /// Creates a `Matrix<T>` with shape `(num_rows, num_columns)`, filled
    /// with ones.
    ///
    /// # Panics
    ///
    /// Panics if `num_rows` or `num_columns` equals zero.
    #[inline]
    pub fn ones(num_rows: usize, num_columns: usize) -> Self
    where
        T: Copy + One,
    {
        Self::full(num_rows, num_columns, T::one())
    }

    /// Returns the number of rows in the matrix.
    #[inline]
    pub const fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Returns the number of columns in the matrix.
    #[inline]
    pub const fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// Returns the shape `(num_rows, num_columns)` of the matrix.
    #[inline]
    pub const fn shape(&self) -> (usize, usize) {
        (self.num_rows, self.num_columns)
    }

    /// Extracts a slice containing the matrix flattened to one dimension.
    #[inline]
    pub fn as_flattened(&self) -> &[T] {
        &self.data
    }

    /// An iterator over the rows of the matrix. The rows are slices.
    ///
    /// As a matrix consists of a sequence of rows, we can iterate through
    /// a matrix by row. This method returns such an iterator.
    pub fn rows(&self) -> Rows<'_, T> {
        Rows::new(&self.data, self.num_columns)
    }

    /// An iterator over the rows of the matrix. The rows are mutable slices.
    ///
    /// As a matrix consists of a sequence of rows, we can iterate through
    /// a matrix by row. This method returns such an iterator.
    pub fn rows_mut(&mut self) -> RowsMut<'_, T> {
        RowsMut::new(&mut self.data, self.num_columns)
    }
}

impl<T, const N: usize, const M: usize> From<[[T; N]; M]> for Matrix<T>
where
    T: Copy,
    [[T; N]; M]: Sized,
{
    /// Creates a `Matrix<T>` with shape `(M, N)` and copies `array`'s items
    /// into it.
    ///
    /// # Panics
    ///
    /// Panics if `M` or `N` equals zero.
    fn from(array: [[T; N]; M]) -> Matrix<T> {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(dimension_name: &str) -> ! {
            panic!("`num_{dimension_name}` (is 0) should be > 0");
        }

        if M == 0 {
            assert_failed("rows");
        }
        if N == 0 {
            assert_failed("columns");
        }
        Self {
            data: array.into_iter().flatten().collect::<Vec<T>>(),
            num_rows: M,
            num_columns: N,
        }
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, row_index: usize) -> &Self::Output {
        let num_columns = self.num_columns;
        &self.data[(row_index * num_columns)..((row_index + 1) * num_columns)]
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, row_index: usize) -> &mut Self::Output {
        let num_columns = self.num_columns;
        &mut self.data[(row_index * num_columns)..((row_index + 1) * num_columns)]
    }
}

#[cfg(test)]
mod test_matrix {
    use super::Matrix;

    #[test]
    fn shape() {
        let num_rows = 3;
        let num_columns = 2;
        let matrix: Matrix<f32> = Matrix::new(num_rows, num_columns);

        assert_eq!(matrix.num_rows(), num_rows);
        assert_eq!(matrix.num_columns(), num_columns);
        assert_eq!(matrix.shape(), (num_rows, num_columns));
    }

    #[test]
    fn full() {
        let num_rows = 2;
        let num_columns = 2;
        let fill_value = 0.5;
        let matrix: Matrix<f32> = Matrix::full(num_rows, num_columns, fill_value);

        for i in 0..num_rows {
            for j in 0..num_columns {
                assert_eq!(matrix[i][j], fill_value);
            }
        }
    }

    #[test]
    #[should_panic(expected = "`num_rows` (is 0) should be > 0")]
    fn full_with_invalid_num_rows() {
        let _: Matrix<f32> = Matrix::full(0, 0, 0.5);
    }

    #[test]
    #[should_panic(expected = "`num_columns` (is 0) should be > 0")]
    fn full_with_invalid_num_columns() {
        let _: Matrix<f32> = Matrix::full(1, 0, 0.5);
    }

    #[test]
    fn new() {
        let num_rows = 2;
        let num_columns = 2;
        let matrix: Matrix<f32> = Matrix::new(num_rows, num_columns);
        let expected_value = Default::default();

        for i in 0..num_rows {
            for j in 0..num_columns {
                assert_eq!(matrix[i][j], expected_value);
            }
        }
    }

    #[test]
    fn zeros() {
        let num_rows = 2;
        let num_columns = 2;
        let matrix: Matrix<f32> = Matrix::zeros(num_rows, num_columns);
        let expected_value = 0.;

        for i in 0..num_rows {
            for j in 0..num_columns {
                assert_eq!(matrix[i][j], expected_value);
            }
        }
    }

    #[test]
    fn ones() {
        let num_rows = 2;
        let num_columns = 2;
        let matrix: Matrix<f32> = Matrix::ones(num_rows, num_columns);
        let expected_value = 1.;

        for i in 0..num_rows {
            for j in 0..num_columns {
                assert_eq!(matrix[i][j], expected_value);
            }
        }
    }

    #[test]
    fn as_flattened() {
        let matrix: Matrix<f32> = Matrix::zeros(2, 2);

        assert_eq!(matrix.as_flattened(), [0., 0., 0., 0.]);
    }

    #[test]
    fn rows() {
        let matrix: Matrix<f32> = Matrix::from([[0., 1.], [2., 3.], [4., 5.]]);
        let mut rows = matrix.rows();

        assert_eq!(rows.next(), Some([0., 1.].as_slice()));
        assert_eq!(rows.next(), Some([2., 3.].as_slice()));
        assert_eq!(rows.next(), Some([4., 5.].as_slice()));
        assert_eq!(rows.next(), None);
    }

    #[test]
    fn rows_mut() {
        let mut matrix: Matrix<f32> = Matrix::from([[0., 1.], [2., 3.], [4., 5.]]);
        let mut rows_mut = matrix.rows_mut();

        let first_row = rows_mut.nth(0).unwrap();
        let last_row = rows_mut.last().unwrap();

        std::mem::swap(&mut first_row[0], &mut last_row[0]);
        std::mem::swap(&mut first_row[1], &mut last_row[1]);

        assert_eq!(matrix, Matrix::from([[4., 5.], [2., 3.], [0., 1.]]));
    }

    #[test]
    fn from() {
        let matrix: Matrix<f32> = Matrix::from([[0., 1., 2.], [3., 4., 5.]]);
    
        assert_eq!(matrix.num_rows(), 2);
        assert_eq!(matrix.num_columns(), 3);
    
        let mut value: f32 = 0.;
        for i in 0..matrix.num_rows() {
            for j in 0..matrix.num_columns() {
                assert_eq!(matrix[i][j], value);
                value += 1.;
            }
        }
    }

    #[test]
    #[should_panic(expected = "`num_rows` (is 0) should be > 0")]
    fn from_with_invalid_num_rows() {
        let _: Matrix<f32> = Matrix::from([[0f32; 3]; 0]);
    }

    #[test]
    #[should_panic(expected = "`num_columns` (is 0) should be > 0")]
    fn from_with_invalid_num_columns() {
        let _: Matrix<f32> = Matrix::from([[0f32; 0]; 2]);
    }

    #[test]
    fn indices() {
        let num_rows = 3;
        let num_columns = 2;
        let mut matrix: Matrix<f32> = Matrix::zeros(num_rows, num_columns);

        // index_mut
        let mut value: f32 = 0.;
        for i in 0..matrix.num_rows() {
            for j in 0..matrix.num_columns() {
                matrix[i][j] = value;
                value += 1.;
            }
        }

        // index
        value = 0.;
        for i in 0..matrix.num_rows() {
            for j in 0..matrix.num_columns() {
                assert_eq!(matrix[i][j], value);
                value += 1.;
            }
        }
    }

    #[test]
    fn partial_eq() {
        let matrix: Matrix<f32> = Matrix::from([[0., 1.], [2., 3.], [4., 5.]]);

        assert_eq!(matrix, matrix);
        assert_eq!(matrix, Matrix::from([[0., 1.], [2., 3.], [4., 5.]]));

        assert_ne!(matrix, Matrix::from([[0., 1., 2.], [3., 4., 5.]]));
        assert_ne!(matrix, Matrix::from([[1., 1.], [2., 3.], [4., 5.]]));
    }
}

#[cfg(doctest)]
mod test_readme {
    macro_rules! external_doc_test {
        ($x:expr) => {
            #[doc = $x]
            extern "C" {}
        };
    }

    external_doc_test!(include_str!("../../README.md"));
}
