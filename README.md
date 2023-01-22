# rust-matrix

An exploration of the Rust language through implementing a matrix library. In this project, I put into practice some importante Rust's features, such as _mutability_, _ownership_, _generic types_, _traits_, and _lifetimes_.

## 1. Creating matrices

This library offers two ways to create a `Matrix<T>`. The most intuitive but least scalable way is to copy items from a two-dimensional array `[[T; N]; M]`. The following code creates a 3-by-2 matrix `Matrix<f32>` from an array `[[f32; 2]; 3]`. The created matrix has 3 rows and 2 columns and its elements are 32-bit floating point numbers.

```rust
use matrix::Matrix;

let a: Matrix<f32> = Matrix::from([[0., 1.], [2., 3.], [4., 5.]]);

assert_eq!(a.num_rows(), 3);
assert_eq!(a.num_columns(), 2);
assert_eq!(a.as_flattened(), [0., 1., 2., 3., 4., 5.]);
```

We can also create a `Matrix<T>` of any size without specifying each of its elements. Instead, the created matrix is filled with copies of a given value of type `T`.

```rust
use matrix::Matrix;

let a = Matrix::full(/* num_rows: */3, /* num_columns: */2, /* fill_value: */0f32);
let b: Matrix<f32> = Matrix::zeros(/* num_rows: */3, /* num_columns: */2);

assert_eq!(a.shape(), (3, 2));
assert_eq!(a, b);
```

In addition to `zeros`, there are also the methods `ones` and `new`: with them, we can create matrices filled with ones or copies of the default value of `T`, respectively.

## 2. Traversing matrices

`Matrix<T>` supports index and mutable index: `a[i][j]` accesses the jth column of the ith row of the matrix `a`. We can also iterate over the rows of a matrix using `Rows` and `RowsMut`: these `Iterator`s are returned by the methods `rows` and `rows_mut` of the `Matrix<T>` objects, respectively.

```rust
use matrix::Matrix;
use std::iter;

let mut a: Matrix<f32> = Matrix::zeros(2, 2);
let b: Matrix<f32> = Matrix::from([[0., 1.], [2., 3.]]);

a[0][1] = 1.;

assert_eq!(a, Matrix::from([[0., 1.], [0., 0.]]));

let mut a_last_row = a.rows_mut().last().unwrap();
let b_last_row = &b[1];

a_last_row.clone_from_slice(b_last_row);

for (a_ith_row, b_ith_row) in iter::zip(a.rows(), b.rows()) {
    assert_eq!(a_ith_row, b_ith_row);
}
```

## 3. Multiplying matrices

This library offers a _CPU cache efficient_ implementation of matrix multiplication that combines _multithreading_ and SIMD (_Single Instruction, Multiple Data_). The process of combining multithreading and SIMD is sometimes called _GPU on CPU_, because GPUs implement a similar technology.

See an example of how to multiply two matrices using this library:

```rust
use matrix::Matrix;

let a: Matrix<f32> = Matrix::from([[0., 1.], [2., 3.], [4., 5.]]);
let b: Matrix<f32> = Matrix::from([[6.], [7.]]);
let c = Matrix::mul(&a, &b);

assert_eq!(c.num_rows(), a.num_rows());
assert_eq!(c.num_columns(), b.num_columns());
assert_eq!(c, Matrix::from([[7.], [33.], [59.]]));
```

As mentioned before, the matrix multiplication of this library uses CPU cache efficiently. CPU cache is based on locality: every time an object is accessed, if it is not already duplicated in the cache, this object and the cache line around it, generally 64 bytes, are transfered into the cache. An unnecessary transfer may even erase from the cache some data needed for subsequent computations, resulting in more unnecessary transfers. To ensure its matrix multiplication is cache efficient, this library implements it in a way that its innermost loop iterates over data stored nearby in memory - or _coalescent_ - for each matrix.

This library uses the [Rayon](https://docs.rs/rayon/latest/rayon/) library to distribute the outermost loop over the available _logical CPU cores_ on the machine runining the operation. Logical cores can be defined as the number of _physical CPU cores_ times the number of threads each one of them can handle through the use of hyperthreading.

SIMD refers to the ability of every logical core in a CPU (or GPU) to apply the same instruction to a vector of data simultaneously. For this reason, it is also called _vectorization_. Contrary to multithreading, SIMD is automatic; it is applied by the compiler "outside of my control". To encourage the compiler to vectorize my implementation of matrix multiplication, I coded its innermost loop in such a way that: it sequentially accesses coalescent data, and it applies the same simple instructions in each iteration (for instance, because of Rust's bounds checking, this implies not using indexes).

For two 1,000-by-1,000 matrices of type `Matrix<f64>`, the matrix multiplication of this library takes 0.144 seconds to execute on my Razer Blade 2015. The version of the Rust compiler is `1.68.0-nightly`. A naive matrix multiplication code, by contrast, takes 3.311 seconds to complete on the same machine: it is about 23x slower. Use `cargo` to reproduce this benchmark on your machine:

```bash
cargo bench -p matrix_multiplication
```