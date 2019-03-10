//! Matrix operations
use rand::prelude::*;

/// A matrix contains its size and its data in columnar format: a vector of columns
/// 
/// Each column is a vector of f64
#[derive(Debug, PartialEq, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    /// create a new scalar: a 1x1 matrix with one value
    pub fn new_scalar(v: f64) -> Matrix {
        Matrix::new(1, 1, &vec![v])
    }

    /// create a new vector: a matrix with one column only
    pub fn new_vector(v: &[f64]) -> Matrix {
        Matrix::new(v.len(), 1, v)
    }

    /// create a matrix of the given size, filled with zeros
    pub fn zero(rows: usize, cols: usize) -> Matrix {
        check(Matrix::all(rows, cols, 0.0))
    }

    /// create a matrix of the given size, filled with the provided value
    pub fn all(rows: usize, cols: usize, v: f64) -> Matrix {
        check(Matrix {
            rows: rows,
            cols: cols,
            data: vec![vec!(v; rows); cols],
        })
    }

    /// create a matrix of the given size, filled with random values
    pub fn rand(rows: usize, cols: usize) -> Matrix {
        let sz = rows * cols;
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..sz).map(|_| {rng.gen()}).collect();
        Matrix::new(rows,cols,&data)
    }

    /// create a matrix of the given size, filled with random values in the given range
    pub fn rand_range(rows: usize, cols: usize, low: f64, high: f64) -> Matrix {
        let sz = rows * cols;
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..sz).map(|_| {rng.gen_range(low,high)}).collect();
        Matrix::new(rows,cols,&data)
    }

    /// create an identity matrix of the given size
    pub fn identity(size: usize) -> Matrix {
        let mut m = Matrix::zero(size, size);
        let mut c = 0;
        for i0 in 0..size {
            m.data[c][i0] = 1.0;
            c += 1;
        }
        check(m)
    }

    /// create a new matrix of the given size, using the given data
    /// the data should be provided in row order: first the first row, then the second row
    pub fn new(rows: usize, cols: usize, data: &[f64]) -> Matrix {
        assert_eq!(rows * cols, data.len());
        let mut m = Matrix::zero(rows, cols);
        for i0 in 0..rows {
            for i1 in 0..cols {
                m.data[i1][i0] = data[i0 * cols + i1];
            }
        }
        check(m)
    }
}

/// get the value at a specific row and column
pub fn get(m: &Matrix, row: usize, col: usize) -> f64{
    assert!(row<m.rows);
    assert!(col<m.cols);
    m.data[col][row]
}

/// set the value at a specific row and column
pub fn set(m: &mut Matrix, row: usize, col: usize, v: f64) {
    assert!(row<m.rows);
    assert!(col<m.cols);
    m.data[col][row]= v;
}

/// get the matrix size
pub fn size(m: &Matrix) -> (usize,usize) {
    (m.rows,m.cols)
}

/// get the matrix data in row order
pub fn get_data(m: &Matrix) -> Vec<f64> {
    let mut v = Vec::with_capacity(m.rows * m.cols);
     for i0 in 0..m.rows {
        for i1 in 0..m.cols {
            v.push(m.data[i1][i0]);
        }
     }
    v
}

/// is the matrix is scalar
pub fn is_scalar(m: &Matrix) -> bool {
    m.rows == 1 && m.cols == 1
}


/// is the matrix a vector
pub fn is_vector(m: &Matrix) -> bool {
    m.cols == 1
}

/// is the matrix an identity matrix
pub fn is_identity(m: &Matrix) -> bool {
    for r in 0..m.rows {
        for c in 0..m.cols {
            if r == c {
                if m.data[c][r]!=1.0 {
                    return false;
                }
            } else {
                if m.data[c][r]!=0.0 {
                    return false;
                }
            }
        }
    }
    true
}

// ensure matrix data is of the right size
fn check(m: Matrix) -> Matrix {
    assert_eq!(m.cols, m.data.len());
    for c in &m.data {
        assert_eq!(m.rows, c.len());
    }
    m
}

/// add a new column with the given data
pub fn add_column(m: &mut Matrix, data: Vec<f64>) {
    assert_eq!(m.rows, data.len());
    m.cols+=1;
    m.data.push(data);
}

/// augment the first matrix with the given one: add all the data as extra columns
/// both matrices need to have the same number of rows
pub fn augment(m1: &mut Matrix, m2: &Matrix) {
    assert_eq!(m1.rows, m2.rows);
    for v in m2.data.iter() {
        m1.data.push(v.clone());
    }
}

/// transpose a matrix
pub fn transpose(m: &Matrix) -> Matrix {
    let mut t = Matrix::zero(m.cols, m.rows);
    for (i0, c) in m.data.iter().enumerate() {
        for (i1, r) in c.iter().enumerate() {
            t.data[i1][i0] = *r;
        }
    }
    t
}

/// have the matrices the same shape?
pub fn same_shape(m1: &Matrix, m2: &Matrix) -> bool {
    m1.rows == m2.rows && m1.cols == m2.cols
}

/// assert matrix ar roughly the same
pub fn assert_eq_aprox(m1: &Matrix, m2: &Matrix) {
    assert!(same_shape(m1,m2));
    for c in 0..m1.cols {
        for r in 0..m1.rows {
            let diff=(m1.data[c][r]-m2.data[c][r]).abs();
            if diff>=0.01{
                println!("{} -> {} : {}", m1.data[c][r],m2.data[c][r],diff);
            }
            assert!(diff < 0.01);
        }
    }
}

/// add the two matrices together
/// they need to have the same shape
pub fn add(m1: &Matrix, m2: &Matrix) -> Matrix {
    assert!(same_shape(m1, m2));
    let mut m = Matrix::zero(m1.rows, m1.cols);
    for i0 in 0..m.cols {
        for i1 in 0..m.rows {
            m.data[i0][i1] = m1.data[i0][i1] + m2.data[i0][i1];
        }
    }
    check(m)
}

/// multiply the matrix by a scalar, returns the new matrix
pub fn mul_scalar(m1: &Matrix, sc: f64) -> Matrix {
    let mut m = Matrix::zero(m1.rows, m1.cols);
    for i0 in 0..m.cols {
        for i1 in 0..m.rows {
            m.data[i0][i1] = m1.data[i0][i1] * sc;
        }
    }
    check(m)
}

/// mutliply the two matrices element wise
/// the matrices need to have the same shape
pub fn mul_element(m1: &Matrix, m2: &Matrix) -> Matrix {
    assert!(same_shape(m1, m2));
    let mut m = Matrix::zero(m1.rows, m1.cols);
    for i0 in 0..m.cols {
        for i1 in 0..m.rows {
            m.data[i0][i1] = m1.data[i0][i1] * m2.data[i0][i1];
        }
    }
    check(m)
}

/// matrix multiplication
pub fn mul(m1: &Matrix, m2: &Matrix) -> Matrix {
    assert_eq!(m1.cols, m2.rows);
    let mut m = Matrix::zero(m1.rows, m2.cols);
    for i0 in 0..m.cols {
        for i1 in 0..m.rows {
            for i2 in 0..m2.rows {
                m.data[i0][i1] += m1.data[i2][i1] * m2.data[i0][i2];
            }
        }
    }
    check(m)
}

/// L2 normalization of a vector
pub fn norm_l2(m1: &Matrix) -> f64 {
    assert!(is_vector(m1));
    f64::sqrt(m1.data[0].iter().map(|x| x.powi(2)).sum())
}

/// L1 normalization of a vector
pub fn norm_l1(m1: &Matrix) -> f64 {
    assert!(is_vector(m1));
    m1.data[0].iter().map(|x| x.abs()).sum()
}

/// normaliza a vector
pub fn norm(m1: &Matrix) -> Matrix {
    let l = norm_l2(m1);
    assert!(l != 0.0);
    Matrix::new_vector(&(m1.data[0].iter().map(|x| x / l).collect::<Vec<f64>>()))
}

/// dot product of a vector
pub fn dot(m1: &Matrix, m2: &Matrix) -> f64 {
    assert!(is_vector(m1));
    assert!(is_vector(m2));
    assert!(same_shape(m1, m2));
    m1.data[0]
        .iter()
        .zip(m2.data[0].iter())
        .map(|(a, b)| a * b)
        .sum()
}

/// swap two rows of the matrix
pub fn row_swap(m1: &mut Matrix, ix0: usize, ix1: usize) {
    assert!(ix0 < m1.rows);
    assert!(ix1 < m1.rows);

    m1.data.iter_mut().for_each(|v| v.swap(ix0, ix1));
}

/// multiply one row of the matrix by a factor
pub fn row_mul(m1: &mut Matrix, ix: usize, sc: f64) {
    assert!(ix < m1.rows);
    assert!(sc != 0.0);
    m1.data.iter_mut().for_each(|v| v[ix] = v[ix] * sc);
}

/// add two rows and multiply by a factor
pub fn row_add(m1: &mut Matrix, ix0: usize, ix1: usize, sc: f64) {
    assert!(ix0 < m1.rows);
    assert!(ix1 < m1.rows);
    assert!(ix0 != ix1);
    m1.data
        .iter_mut()
        .for_each(|v| v[ix0] = v[ix0] + v[ix1] * sc);
}

/// gaussian elimination
pub fn gauss_elim(m1: &mut Matrix) {
    let mut r = 0;
    let mut c = 0;
    while r < m1.rows && c < m1.cols {
        let mut i_max = r;
        let mut v_max = m1.data[c][i_max].abs();
        for r2 in r + 1..m1.rows {
            if m1.data[c][r2].abs() > v_max {
                i_max = r2;
                v_max = m1.data[c][i_max].abs();
            }
        }
        if m1.data[c][i_max] == 0.0 {
            c += 1;
        } else {
            row_swap(m1, r, i_max);
            for i in r + 1..m1.rows {
                let f = m1.data[c][i] / m1.data[c][r];
                m1.data[c][i] = 0.0;
                for j in c + 1..m1.cols {
                    m1.data[j][i] = m1.data[j][i] - m1.data[j][r] * f;
                }
            }
            r += 1;
            c += 1;
        }
    }
}

/// Gauss/Jordan elimination
pub fn gauss_jordan_elim(m1: &mut Matrix) {
    let mut c = 0;
    let mut r = 0;
    while r<m1.rows && c<m1.cols {
        for r1 in r..m1.rows {
            if m1.data[c][r1] != 0.0 {
                if r1 > r {
                    row_swap(m1, r, r1);
                }
                row_mul(m1, r, 1.0 / m1.data[c][r]);
                for r1 in 0..m1.rows {
                    if r1 != r {
                        row_add(m1, r1, r, -m1.data[c][r1]);
                    }
                }
                r+=1;
                break;
            }
        }
        c+=1;
    }
}

/// matrix reduction
pub fn reduce(m1: &mut Matrix) {
    for r in 0..m1.rows {
        let mut c = 0;
        while c < m1.cols && m1.data[c][r] == 0.0 {
            c += 1;
        }
        if c < m1.cols {
            row_mul(m1, r, 1.0 / m1.data[c][r]);
        }
    }
}

/// matrix inverse
pub fn inverse(m: &mut Matrix) -> Matrix {
    augment(m, &Matrix::identity(m.rows));
    gauss_jordan_elim(m);
    let mut ret=Matrix::zero(m.rows,m.cols);
    for c in m.cols..m.cols*2 {
        for r in 0..m.rows {
            ret.data[c-m.cols][r]=m.data[c][r];
        }
    }
    ret
}

/// apply a function to all the data in the matrix
pub fn map(m: &mut Matrix, f: fn(f64) -> f64) {
    for i0 in 0..m.cols {
        for i1 in 0..m.rows {
            m.data[i0][i1]=f(m.data[i0][i1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar() {
        let sc = Matrix::new_scalar(3.14);
        assert!(is_scalar(&sc));
        assert!(is_scalar(&Matrix::zero(1, 1)));
        assert_eq!(sc, transpose(&sc));
    }

    #[test]
    fn test_vector() {
        let v = Matrix::new_vector(&vec![1.0, 2.0, 3.14]);
        assert!(is_vector(&v));
        assert!(is_vector(&Matrix::zero(3, 1)));
        let m = Matrix::new(1, 3, &vec![1.0, 2.0, 3.14]);
        assert_eq!(m, transpose(&v));
    }

    #[test]
    fn test_matrix() {
        let m1 = Matrix::new(3, 2, &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(0.0, m1.data[0][0]);
        assert_eq!(2.0, m1.data[0][1]);
        assert_eq!(4.0, m1.data[0][2]);
        assert_eq!(1.0, m1.data[1][0]);
        assert_eq!(3.0, m1.data[1][1]);
        assert_eq!(5.0, m1.data[1][2]);
        assert_eq!(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],get_data(&m1));
        let m2 = Matrix::new(2, 3, &vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
        assert_eq!(m2, transpose(&m1));

    }

    #[test]
    fn test_add() {
        let m1 = Matrix::new(2, 2, &vec![0.0, 1.0, 2.0, 3.0]);
        let m2 = Matrix::all(2, 2, 1.0);
        let m3 = Matrix::new(2, 2, &vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m3, add(&m1, &m2));
    }

    #[test]
    fn test_mul_scalar() {
        let m1 = Matrix::new(2, 2, &vec![0.0, 1.0, 2.0, 3.0]);
        let m2 = Matrix::new(2, 2, &vec![0.0, 2.0, 4.0, 6.0]);
        assert_eq!(m2, mul_scalar(&m1, 2.0));
    }

    #[test]
    fn test_mul_element() {
        let m1 = Matrix::new(2, 2, &vec![0.0, 1.0, 2.0, 3.0]);
        let m2 = Matrix::new(2, 2, &vec![0.0, 1.0, 4.0, 9.0]);
        assert_eq!(m2, mul_element(&m1, &m1));
    }

    #[test]
    fn test_mul1() {
        // https://www.mathsisfun.com/algebra/matrix-multiplying.html
        let m1 = Matrix::new(2, 3, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m2 = Matrix::new(3, 2, &vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let m3 = Matrix::new(2, 2, &vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(m3, mul(&m1, &m2));
    }

    #[test]
    fn test_mul2() {
        // https://www.mathsisfun.com/algebra/matrix-multiplying.html
        let m1 = Matrix::new(1, 3, &vec![3.0, 4.0, 2.0]);
        let m2 = Matrix::new(
            3,
            4,
            &vec![13.0, 9.0, 7.0, 15.0, 8.0, 7.0, 4.0, 6.0, 6.0, 4.0, 0.0, 3.0],
        );
        let m3 = Matrix::new(1, 4, &vec![83.0, 63.0, 37.0, 75.0]);
        assert_eq!(m3, mul(&m1, &m2));
    }

    #[test]
    fn test_mul3() {
        // https://medium.com/@LeonFedden/a-hackers-guide-to-deep-learnings-secret-sauces-linear-algebra-555403c3be16
        let m1 = Matrix::new(
            3,
            4,
            &vec![
                0.21922347, 0.84313988, 0.41381942, 0.53553901, 0.35322431, 0.38337327, 0.15964194,
                0.30629508, 0.16188791, 0.55971721, 0.33561351, 0.04709838,
            ],
        );
        let m2 = Matrix::new(
            3,
            3,
            &vec![
                1.2169923306461716,
                0.6307682507191089,
                0.6715159385047504,
                0.6307682507191089,
                0.39104470236463895,
                0.33976735627664856,
                0.6715159385047504,
                0.33976735627664856,
                0.4543457360674967,
            ],
        );
        assert_eq!(m2, mul(&m1, &transpose(&m1)));
    }

    #[test]
    fn test_identity() {
        let id1 = Matrix::identity(1);
        assert_eq!(1.0, id1.data[0][0]);
        assert!(is_identity(&id1));
        let id3 = Matrix::identity(3);
        assert_eq!(1.0, id3.data[0][0]);
        assert_eq!(1.0, id3.data[1][1]);
        assert_eq!(1.0, id3.data[2][2]);
        assert_eq!(0.0, id3.data[0][1]);
        assert_eq!(0.0, id3.data[0][2]);
        assert_eq!(0.0, id3.data[1][0]);
        assert_eq!(0.0, id3.data[1][2]);
        assert_eq!(0.0, id3.data[2][0]);
        assert_eq!(0.0, id3.data[2][1]);
        assert!(is_identity(&id3));
        let v = Matrix::new_vector(&vec![1.0, 2.0, 3.14]);
        assert_eq!(v, mul(&Matrix::identity(3), &v));
        assert_eq!(v, mul(&v, &Matrix::identity(1)));
    }

    #[test]
    fn test_norm() {
        let v1 = Matrix::new_vector(&vec![0.8, 0.8]);
        assert_eq!(1.1313708498984762, norm_l2(&v1));
        assert_eq!(1.6, norm_l1(&v1));
    }

    #[test]
    fn test_dot() {
        let v1 = Matrix::new_vector(&vec![1.0, 3.0, -5.0]);
        let v2 = Matrix::new_vector(&vec![4.0, -2.0, -1.0]);
        assert_eq!(3.0, dot(&v1, &v2));
        assert_eq!(Matrix::new_scalar(3.0), mul(&Matrix::new(1, 3, &v2.data[0]), &v1));
    }

    #[test]
    fn test_row() {
        let mut m1 = Matrix::new(3, 2, &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let m2 = Matrix::new(3, 2, &vec![4.0, 5.0, 2.0, 3.0, 0.0, 1.0]);
        row_swap(&mut m1, 0, 2);
        assert_eq!(m2, m1);
        let m3 = Matrix::new(3, 2, &vec![4.0, 5.0, 4.0, 6.0, 0.0, 1.0]);
        row_mul(&mut m1, 1, 2.0);
        assert_eq!(m3, m1);
        let m4 = Matrix::new(3, 2, &vec![12.0, 17.0, 4.0, 6.0, 0.0, 1.0]);
        row_add(&mut m1, 0, 1, 2.0);
        assert_eq!(m4, m1);
    }
    #[test]
    fn test_gauss_elim() {
        // https://www.geeksforgeeks.org/gaussian-elimination/
        let mut m1 = Matrix::new(
            3,
            4,
            &vec![
                3.0, 2.0, -4.0, 3.0, 2.0, 3.0, 3.0, 15.0, 5.0, -3.0, 1.0, 14.0,
            ],
        );
        gauss_elim(&mut m1);
        let m2 = Matrix::new(
            3,
            4,
            &vec![
                5.0, -3.0, 1.0, 14.0, 0.0, 4.2, 2.6, 9.4, 0.0, 0.0, -6.95, -13.9,
            ],
        );
        assert_eq_aprox(&m2, &m1);
    }

    #[test]
    fn test_gauss_jordan_elim() {
        // https://en.wikipedia.org/wiki/Gaussian_elimination
        let mut m1 = Matrix::new(3, 4, &vec![2.0, 1.0, -1.0, 8.0,
        -3.0, -1.0, 2.0, -11.0,
        -2.0, 1.0, 2.0, -3.0]);
        gauss_jordan_elim(&mut m1);
        let m2 = Matrix::new(3, 4, &vec![1.0, 0.0, 0.0, 2.0,
        0.0, 1.0, 0.0, 3.0,
        0.0, 0.0, 1.0, -1.0]);
       
        assert_eq!(m2, m1);
    }

    #[test]
    fn test_gauss_jordan_elim2() {
        // https://www.statlect.com/matrix-algebra/Gauss-Jordan-elimination
        let mut m1 = Matrix::new(3, 5, &vec![-1.0, 2.0, 6.0, 7.0, 15.0,
        3.0, -6.0, 0.0, -3.0, -9.0,
        1.0, 0.0, 6.0, -1.0, 5.0]);
        gauss_jordan_elim(&mut m1);
        let m2 = Matrix::new(3, 5, &vec![1.0, 0.0, 0.0, -7.0, -7.0,
        0.0, 1.0, 0.0, -3.0, -2.0,
        0.0, 0.0, 1.0, 1.0, 2.0]);
       
        assert_eq!(m2, m1);
    }

    #[test]
    fn test_inverse(){
        let mut m1 = Matrix::new(3,3,&vec!(2.0,-1.0,0.0,
            -1.0,2.0,-1.0,
            0.0,-1.0,2.0
            ));
        let m2 = inverse(&mut m1);
        let m3 = Matrix::new(3,3,&vec!(3.0/4.0,1.0/2.0,1.0/4.0,
            1.0/2.0,1.0,1.0/2.0,
            1.0/4.0,1.0/2.0,3.0/4.0
            ));
        assert_eq_aprox(&m3,&m2);
    }

     #[test]
    fn test_map() {
        let mut m1 = Matrix::new(3, 2, &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        map(&mut m1,|v| v*2.0);
        assert_eq!(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0],get_data(&m1));
    }
}
