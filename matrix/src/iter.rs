/// An iterator over the rows of a matrix.
///
/// This struct is created by the [`rows`] method on [`Matrix`].
/// See its documentation for more.
///
/// [`rows`]: super::Matrix::rows
/// [`Matrix`]: super::Matrix
#[derive(Clone, Debug)]
pub struct Rows<'a, T: 'a> {
    slice: &'a [T],
    num_columns: usize,
}

impl<'a, T: 'a> Rows<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a [T], num_columns: usize) -> Self {
        Self { slice, num_columns }
    }

    #[inline]
    fn len(&self) -> usize {
        self.slice.len() / self.num_columns
    }
}

impl<'a, T> Iterator for Rows<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let (head, tail) = self.slice.split_at(self.num_columns);
            self.slice = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.is_empty() {
            (0, Some(0))
        } else {
            let result = self.len();
            (result, Some(result))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n > self.len() {
            self.slice = &[];
            None
        } else {
            let start = n * self.num_columns;
            let end = start + self.num_columns;
            let nth = &self.slice[start..end];
            self.slice = &self.slice[end..];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let start = self.slice.len() - self.num_columns;
            let last = &self.slice[start..];
            Some(last)
        }
    }
}

/// An iterator over the mutable rows of a matrix.
///
/// This struct is created by the [`rows_mut`] method on [`Matrix`].
/// See its documentation for more.
///
/// [`rows`]: super::Matrix::rows
/// [`Matrix`]: super::Matrix
#[derive(Debug)]
pub struct RowsMut<'a, T: 'a> {
    slice: &'a mut [T],
    num_columns: usize,
}

impl<'a, T: 'a> RowsMut<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a mut [T], num_columns: usize) -> Self {
        Self { slice, num_columns }
    }

    #[inline]
    fn len(&self) -> usize {
        self.slice.len() / self.num_columns
    }
}

impl<'a, T> Iterator for RowsMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let slice = std::mem::take(&mut self.slice);
            let (head, tail) = slice.split_at_mut(self.num_columns);
            self.slice = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.is_empty() {
            (0, Some(0))
        } else {
            let result = self.len();
            (result, Some(result))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n > self.len() {
            self.slice = &mut [];
            None
        } else {
            let start = n * self.num_columns;
            let end = start + self.num_columns;

            let slice = std::mem::take(&mut self.slice);
            let (head, tail) = slice.split_at_mut(end);
            self.slice = tail;

            let (_, nth) = head.split_at_mut(start);
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let start = self.slice.len() - self.num_columns;
            let slice = self.slice;
            let (_, last) = slice.split_at_mut(start);
            Some(last)
        }
    }
}

#[cfg(test)]
mod test_rows {
    use super::Rows;

    #[test]
    fn next() {
        let data = [0, 1, 2, 3, 4, 5];
        let num_columns = 2;
        let mut rows = Rows::new(&data, num_columns);

        assert_eq!(rows.next(), Some([0, 1].as_slice()));
        assert_eq!(rows.next(), Some([2, 3].as_slice()));
        assert_eq!(rows.next(), Some([4, 5].as_slice()));

        assert_eq!(rows.next(), None);
    }

    #[test]
    fn size_hint() {
        let data = [0, 1, 2, 3, 4, 5];
        let num_rows = 3;
        let num_columns = 2;
        let rows = Rows::new(&data, num_columns);

        assert_eq!(data.len(), num_rows * num_columns);
        assert_eq!(rows.size_hint(), (num_rows, Some(num_rows)));
    }

    #[test]
    fn count() {
        let data = [0, 1, 2, 3, 4, 5];
        let num_rows = 3;
        let num_columns = 2;
        let rows = Rows::new(&data, num_columns);

        assert_eq!(data.len(), num_rows * num_columns);
        assert_eq!(rows.count(), num_rows);
    }

    #[test]
    fn nth() {
        let data = [0, 1, 2, 3, 4, 5];
        let num_columns = 2;
        let mut rows = Rows::new(&data, num_columns);

        assert_eq!(rows.nth(1), Some([2, 3].as_slice()));
        assert_eq!(rows.next(), Some([4, 5].as_slice()));
    }

    #[test]
    fn last() {
        let data = [0, 1, 2, 3, 4, 5];
        let num_columns = 2;
        let rows = Rows::new(&data, num_columns);

        assert_eq!(rows.last(), Some([4, 5].as_slice()));
    }
}

#[cfg(test)]
mod test_rows_mut {
    use super::RowsMut;

    #[test]
    fn next() {
        let mut data = [0, 1, 2, 3, 4, 5];
        let num_columns = 2;
        let mut rows_mut = RowsMut::new(&mut data, num_columns);

        assert_eq!(rows_mut.next(), Some([0, 1].as_mut_slice()));
        assert_eq!(rows_mut.next(), Some([2, 3].as_mut_slice()));
        assert_eq!(rows_mut.next(), Some([4, 5].as_mut_slice()));

        assert_eq!(rows_mut.next(), None);
    }

    #[test]
    fn mutability() {
        let mut data = [0, 1, 2, 3, 4, 5];
        let num_columns = 2;
        let mut rows_mut = RowsMut::new(&mut data, num_columns);

        let first_row = rows_mut.nth(0).unwrap();
        let last_row = rows_mut.last().unwrap();
        std::mem::swap(&mut first_row[0], &mut last_row[0]);
        std::mem::swap(&mut first_row[1], &mut last_row[1]);

        assert_eq!(data, [4, 5, 2, 3, 0, 1]);
    }

    #[test]
    fn size_hint() {
        let mut data = [0, 1, 2, 3, 4, 5];
        let num_rows = 3;
        let num_columns = 2;

        assert_eq!(data.len(), num_rows * num_columns);

        let rows_mut = RowsMut::new(&mut data, num_columns);

        assert_eq!(rows_mut.size_hint(), (num_rows, Some(num_rows)));
    }

    #[test]
    fn count() {
        let mut data = [0, 1, 2, 3, 4, 5];
        let num_rows = 3;
        let num_columns = 2;

        assert_eq!(data.len(), num_rows * num_columns);

        let rows_mut = RowsMut::new(&mut data, num_columns);

        assert_eq!(rows_mut.count(), num_rows);
    }

    #[test]
    fn nth() {
        let mut data = [0, 1, 2, 3, 4, 5];
        let num_columns = 2;
        let mut rows_mut = RowsMut::new(&mut data, num_columns);

        assert_eq!(rows_mut.nth(1), Some([2, 3].as_mut_slice()));
        assert_eq!(rows_mut.next(), Some([4, 5].as_mut_slice()));
    }

    #[test]
    fn last() {
        let mut data = [0, 1, 2, 3, 4, 5];
        let num_columns = 2;
        let rows_mut = RowsMut::new(&mut data, num_columns);

        assert_eq!(rows_mut.last(), Some([4, 5].as_mut_slice()));
    }
}
