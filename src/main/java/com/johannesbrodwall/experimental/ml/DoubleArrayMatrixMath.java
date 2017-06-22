package com.johannesbrodwall.experimental.ml;

public class DoubleArrayMatrixMath {

    public void divide(double[][] A, double[][] B, double[][] result) {
        check(cols(A), 1, "A[" + shape(A) + "].cols != 1");
        check(rows(A), rows(B), "A[" + shape(A) + "].rows != B[" + shape(B) + "].rows");
        checkShape(B, result, "B", "result");

        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[0].length; col++) {
                result[row][col] = A[row][0] / B[row][col];
            }
        }
    }

    /** A[r][c] + scale * B[r][c] => result[r][c] */
    public void add(double[][] A, double scale, double[][] B, double[][] result) {
        checkShape(A, result, "A", "result");
        checkShape(B, result, "B", "result");
        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[0].length; col++) {
                result[row][col] = A[row][col] + scale * B[row][col];
            }
        }
    }

    private void checkShape(double[][] a, double[][] b, String aName, String bName) {
        check(rows(a), rows(b), aName + "[" + shape(a) + "] != " + bName + "[" + shape(b) + "]");
        check(cols(a), cols(b), aName + "[" + shape(a) + "] != " + bName + "[" + shape(b) + "]");
    }

    private void check(int a, int b, String message) {
        if (a != b) {
            throw new IllegalArgumentException(message);
        }
    }

    private String shape(double[][] a) {
        return rows(a) + "x" + cols(a);
    }

    private int rows(double[][] m) {
        return m.length;
    }

    private int cols(double[][] m) {
        return m[0].length;
    }

    public double[][] transpose(double[][] values) {
        double[][] result = new double[values[0].length][values.length];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[i].length; j++) {
                result[i][j] = values[j][i];
            }
        }
        return result;
    }


}
