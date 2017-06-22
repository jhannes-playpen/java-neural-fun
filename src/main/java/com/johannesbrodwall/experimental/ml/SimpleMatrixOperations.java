package com.johannesbrodwall.experimental.ml;

public class SimpleMatrixOperations implements MatrixOperations {

    private DoubleArrayMatrixMath math = new DoubleArrayMatrixMath();

    private class SimpleMatrix implements Matrix {

        private double[][] values;

        public SimpleMatrix(double[][] values) {
            this.values = values;
        }

        public int rows() {
            return values.length;
        }

        public int cols() {
            return values[0].length;
        }

        public double get(int row, int col) {
            return values[row][col];
        }

        @Override
        public String toString() {
            return getClass().getSimpleName() + "{" + shape() + "}";
        }

        private String shape() {
            return rows() + "x" + cols();
        }

    }

    @Override
    public Matrix of(double[][] values) {
        return new SimpleMatrix(values);
    }

    @Override
    public Matrix ofTransposed(double[][] values) {
        return new SimpleMatrix(math.transpose(values));
    }

    @Override
    public Matrix add(Matrix Agen, Matrix Bgen) {
        double[][] A = unwrap(Agen);
        double[][] result = new double[A.length][A[0].length];
        math.add(A, 1.0, unwrap(Bgen), result);
        return wrap(result);
    }

    @Override
    public Matrix subtract(Matrix Agen, Matrix Bgen) {
        double[][] A = unwrap(Agen);
        double[][] result = new double[A.length][A[0].length];
        math.add(A, -1.0, unwrap(Bgen), result);
        return wrap(result);
    }

    @Override
    public Matrix multiplyElements(Matrix Agen, Matrix Bgen) {
        SimpleMatrix A = (SimpleMatrix)Agen;
        SimpleMatrix B = (SimpleMatrix)Bgen;

        check(A.rows(), B.rows(), "A.rows != B.rows");
        check(A.cols(), B.cols(), "A.cols != B.cols");
        double[][] result = new double[A.rows()][A.cols()];
        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[0].length; col++) {
                for (int i=0; i<A.rows(); i++) {
                    result[row][col] = A.get(row, col) * B.get(row, col);
                }
            }
        }
        return new SimpleMatrix(result);
    }

    /** result[n x p] = (A[n x m] * B[m x p]) + (C[1 x p] x ROW[n]), where C has 1 column
    /** result[n x p] = (A[n x m] * B[m x p]) + (C[n x p]), where C has 1 column.
     *
     * result[a][b] = C[0][b] + sum(0..n) { A[a][i] * B[i][b] }
     */
    @Override
    public Matrix multiplyAdd(Matrix Agen, Matrix Bgen, Matrix Cgen) {
        SimpleMatrix A = (SimpleMatrix)Agen;
        SimpleMatrix B = (SimpleMatrix)Bgen;
        SimpleMatrix C = (SimpleMatrix)Cgen;

        check(A.cols(), B.rows(), "A[" + A.shape() + "].cols != B[" + B.shape() + "].rows");
        check(B.cols(), C.cols(), "B[" + B.shape() + "].cols != C[" + C.shape() + "].cols");
        check(C.rows(), 1, "C[" + C.shape() + "].rows() != 1");

        double[][] result = new double[A.rows()][B.cols()];
        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[0].length; col++) {
                for (int i=0; i<A.cols(); i++) {
                    result[row][col] += A.get(row, i) * B.get(i, col);
                }
                result[row][col] += C.get(0, col);
            }
        }

        return new SimpleMatrix(result);
    }

    @Override
    /** result[n x p] = (A[n x m] * B[m x p]).
     * result[a][b] = sum(0..n) { A[a][i] * B[i][b] }
     *  */
    public Matrix multiply(Matrix Agen, Matrix Bgen) {
        SimpleMatrix A = (SimpleMatrix)Agen;
        SimpleMatrix B = (SimpleMatrix)Bgen;
        check(A.cols(), B.rows(), "A[" + A.shape() + "].rows != B[" + B.shape() + "].cols");

        double[][] result = new double[A.rows()][B.cols()];
        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[0].length; col++) {
                for (int i=0; i<A.cols(); i++) {
                    result[row][col] += A.get(row, i) * B.get(i, col);
                }
            }
        }

        return new SimpleMatrix(result);
    }

    /** result[1 x n] = A[1 x 1] / B[1 x n]
     *  result[0,a]  = A[0,0] / B[0, a]
     */
    @Override
    public Matrix divide(Matrix Agen, Matrix Bgen) {
        SimpleMatrix A = (SimpleMatrix)Agen;
        SimpleMatrix B = (SimpleMatrix)Bgen;


        check(A.cols(), 1, "A[" + A.shape() + "]");

        if (B.rows() == 1) {
            double[][] result = new double[A.rows()][B.cols()];
            for (int row = 0; row < result.length; row++) {
                for (int col = 0; col < result[0].length; col++) {
                    result[row][col] = A.get(row, 0) / B.get(0, col);
                }
            }

            return new SimpleMatrix(result);
        }

        check(A.rows(), B.rows(), "A[" + A.shape() + "].rows != B[" + B.shape() + "].rows");

        double[][] result = new double[B.rows()][B.cols()];
        math.divide(A.values, B.values, result);
        return new SimpleMatrix(result);
    }

    @Override
    public Matrix divideTransposed(Matrix Agen, Matrix Bgen) {
        SimpleMatrix A = (SimpleMatrix)Agen;
        SimpleMatrix B = (SimpleMatrix)Bgen;

        check(A.cols(), 1, "A[" + A.shape() + "]");

        if (B.cols() == 1) {
            double[][] result = new double[A.rows()][B.rows()];
            for (int row = 0; row < result.length; row++) {
                for (int col = 0; col < result[0].length; col++) {
                    result[row][col] = A.get(row, 0) / B.get(col, 0);
                }
            }

            return new SimpleMatrix(result);
        } else {
            throw new IllegalArgumentException("Ugh!");
        }
    }

    @Override
    public double[][] values(Matrix m) {
        return ((SimpleMatrix)m).values;
    }

    @Override
    public Matrix apply(Matrix matrix, Function f) {
        double[][] input = ((SimpleMatrix)matrix).values;
        double[][] output = new double[input.length][input[0].length];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[i].length; j++) {
                output[i][j] = f.apply(input[i][j]);
            }
        }
        return new SimpleMatrix(output);
    }

    @Override
    public Matrix transpose(Matrix A) {
        return wrap(math.transpose(unwrap(A)));
    }

    private Matrix wrap(double[][] values) {
        return new SimpleMatrix(values);
    }

    private double[][] unwrap(Matrix m) {
        return ((SimpleMatrix)m).values;
    }

    private void check(int a, int b, String message) {
        if (a != b) {
            throw new IllegalArgumentException(message);
        }
    }
}
