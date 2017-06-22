package com.johannesbrodwall.experimental.ml;

import no.uib.cipr.matrix.DenseMatrix;

public class CiprMatrixOperations implements MatrixOperations {

    private DoubleArrayMatrixMath math = new DoubleArrayMatrixMath();

    private static class MatrixWrapper implements Matrix {
        private DenseMatrix denseMatrix;

        public MatrixWrapper(DenseMatrix denseMatrix) {
            this.denseMatrix = denseMatrix;
        }
    }

    @Override
    public Matrix of(double[][] values) {
        return new MatrixWrapper(new DenseMatrix(values));
    }

    @Override
    public Matrix ofTransposed(double[][] values) {
        DenseMatrix m = new DenseMatrix(values);
        DenseMatrix transposed = new DenseMatrix(m.numColumns(), m.numRows());
        m.transpose(transposed);
        return new MatrixWrapper(transposed);
    }

    @Override
    public Matrix multiply(Matrix Agen, Matrix Bgen) {
        DenseMatrix A = unwrap(Agen);
        DenseMatrix B = unwrap(Bgen);
        DenseMatrix result = new DenseMatrix(A.numRows(), B.numColumns());
        A.mult(B, result);

        return wrap(result);
    }

    @Override
    public Matrix divide(Matrix Agen, Matrix Bgen) {
        DenseMatrix A = unwrap(Agen);
        DenseMatrix B = unwrap(Bgen);
        DenseMatrix m = new DenseMatrix(A.numRows(), B.numColumns());
        if (B.numRows() == 1) {

            for (int row = 0; row < m.numRows(); row++) {
                for (int col = 0; col < m.numColumns(); col++) {
                    m.set(row, col, A.get(row, 0) / B.get(0, col));
                }
            }
            return wrap(m);
        } else {
            double[][] result = new double[B.numRows()][B.numColumns()];
            for (int row = 0; row < result.length; row++) {
                for (int col = 0; col < result[0].length; col++) {
                    result[row][col] = A.get(row, 0) / B.get(row, col);
                }
            }
            return wrap(result);
        }
    }

    @Override
    public Matrix divideTransposed(Matrix Agen, Matrix Bgen) {
        DenseMatrix A = unwrap(Agen);
        DenseMatrix B = unwrap(Bgen);

        if (B.numColumns() == 1) {
            double[][] result = new double[A.numRows()][B.numRows()];
            for (int row = 0; row < result.length; row++) {
                for (int col = 0; col < result[0].length; col++) {
                    result[row][col] = A.get(row, 0) / B.get(col, 0);
                }
            }

            return wrap(result);
        } else {
            throw new IllegalArgumentException("Ugh!");
        }
    }

    private Matrix wrap(double[][] values) {
        return wrap(new DenseMatrix(values));
    }

    @Override
    public Matrix add(Matrix A, Matrix B) {
        return wrap(unwrap(A).add(unwrap(B).copy()));
    }

    @Override
    public Matrix subtract(Matrix A, Matrix B) {
        return wrap(unwrap(A).add(-1, unwrap(B).copy()));
    }

    @Override
    public Matrix multiplyElements(Matrix Agen, Matrix Bgen) {
        DenseMatrix A = unwrap(Agen);
        DenseMatrix B = unwrap(Bgen);
        DenseMatrix m = new DenseMatrix(A.numRows(), B.numRows());
        for (int i = 0; i < m.numRows(); i++) {
            for (int j = 0; j < m.numColumns(); j++) {
                m.set(i, j, A.get(i, j) * B.get(i, j));
            }
        }
        return wrap(m);
    }

    @Override
    public Matrix multiplyAdd(Matrix Agen, Matrix Bgen, Matrix Cgen) {
        DenseMatrix A = unwrap(Agen);
        DenseMatrix B = unwrap(Bgen);
        DenseMatrix C = unwrap(Cgen);

        DenseMatrix result;

        if (C.numRows() == 1) {
            result = new DenseMatrix(A.numRows(), C.numColumns());
            for (int row = 0; row < result.numRows(); row++) {
                for (int col = 0; col < result.numColumns(); col++) {
                    result.set(row, col, C.get(0, col));
                }
            }
        } else {
            result = C.copy();
        }

        A.multAdd(B, result);
        return new MatrixWrapper(result);
    }

    @Override
    public double[][] values(Matrix wrapped) {
        DenseMatrix m = unwrap(wrapped);
        double[][] output = new double[m.numRows()][m.numColumns()];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[i].length; j++) {
                output[i][j] = m.get(i, j);
            }
        }
        return output;
    }

    @Override
    public Matrix apply(Matrix wrapped, Function f) {
        DenseMatrix m = unwrap(wrapped).copy();
        for (int i = 0; i < m.numRows(); i++) {
            for (int j = 0; j < m.numColumns(); j++) {
                m.set(i, j, f.apply(m.get(i, j)));
            }
        }
        return wrap(m);
    }

    @Override
    public Matrix transpose(Matrix Agen) {
        DenseMatrix A = unwrap(Agen);
        DenseMatrix B = new DenseMatrix(A.numColumns(), A.numRows());
        A.transpose(B);
        return new MatrixWrapper(B);
    }

    private MatrixWrapper wrap(no.uib.cipr.matrix.Matrix m) {
        return new MatrixWrapper((DenseMatrix)m);
    }

    private DenseMatrix unwrap(Matrix m) {
        return ((MatrixWrapper)m).denseMatrix;
    }

}
