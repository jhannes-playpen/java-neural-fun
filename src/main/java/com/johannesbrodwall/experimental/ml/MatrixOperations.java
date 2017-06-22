package com.johannesbrodwall.experimental.ml;

public interface MatrixOperations {

    public interface Function {
        double apply(double value);
    }

    Matrix of(double[][] values);

    Matrix ofTransposed(double[][] values);

    /** result[n x p] = (A[n x m] * B[m x p]) + (C[1 x p] x ROW[n]), where C has 1 column
    /** result[n x p] = (A[n x m] * B[m x p]) + (C[n x p]), where C has 1 column
     */
    Matrix multiplyAdd(Matrix A, Matrix B, Matrix C);

    double[][] values(Matrix m);

    Matrix apply(Matrix matrix, Function f);

    /** result[n x p] = (A[n x m] * B[m x p]). The resulting Matrix has dimensions {A.rows, B.cols}.
     * TODO: Result arg instead of return value in order to cache matrices
     *  If A is an n × m matrix and B is an m × p matrix, their matrix product AB is an n × p matrix
     *  */
    Matrix multiply(Matrix A, Matrix B);

    Matrix divide(Matrix A, Matrix B);

    Matrix divideTransposed(Matrix A, Matrix B_T);

    Matrix add(Matrix outputWeights, Matrix transpose);

    Matrix subtract(Matrix A, Matrix B);

    Matrix multiplyElements(Matrix A, Matrix B);

    Matrix transpose(Matrix A);





}
