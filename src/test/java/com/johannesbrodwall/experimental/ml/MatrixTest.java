package com.johannesbrodwall.experimental.ml;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.Test;

import com.johannesbrodwall.experimental.ml.Matrix;
import com.johannesbrodwall.experimental.ml.MatrixOperations;

public abstract class MatrixTest {

    @Test
    public void shouldMultiplyAdd() {
        Matrix A = matrix().of(new double[][] {
            { 2, 3, 5 },
            { 7, 11, 13 },
            { 17, 23, 29 },
            { 31, 37, 43 },
        });
        Matrix B = matrix().of(new double[][] {
            { 100, 1000 },
            { 10_000, 100_000 },
            { 1_000_000, 10_000_0000 },
        });
        Matrix C = matrix().of(new double[][] {
            { 1, 2 }
        });

        double[][] result = matrix().values(matrix().multiplyAdd(A, B, C));
        assertThat(result).hasSize(4);
        assertThat(result[0]).hasSize(2);
        assertThat(result).isEqualTo(new double[][] {
            {
                100*2 + 10_000*3 + 1_000_000*5 + 1,
                1000*2 + 100_000*3 + 100_000_000*5 + 2,
            },
            {
                100*7 + 10_000*11 + 1_000_000*13 + 1,
                1000*7 + 100_000*11 + 100_000_000*13 + 2,
            },
            {
                100.0*17 + 10_000.0*23 + 1_000_000.0*29 + 1,
                1000.0*17 + 100_000.0*23 + 100_000_000.0*29 + 2,
            },
            {
                100.0*31 + 10_000.0*37 + 1_000_000.0*43 + 1,
                1000.0*31 + 100_000.0*37 + 100_000_000.0*43 + 2,
            },
        });
    }

    @Test
    public void shouldMultiply() {
        Matrix A = matrix().of(new double[][] {
            { 100,              1000,           10_000 },
            { 100_000,          1_000_000,      10_000_000 },
        });
        Matrix B = matrix().of(new double[][] {
            { 2, 3 },
            { 5, 7 },
            { 11, 13 },
        });

        double[][] result = matrix().values(matrix().multiply(A, B));
        assertThat(result).isEqualTo(new double[][] {
            {
                2 * 100         + 5 * 1000      + 11 * 10_000,
                3 * 100         + 7 * 1000      + 13 * 10_000,
            },
            {
                2 * 100_000     + 5 * 1_000_000 + 11 * 10_000_000,
                3 * 100_000     + 7 * 1_000_000 + 13 * 10_000_000,
            }
        });
    }

    @Test
    public void columnVector() {
        Matrix A = matrix().of(new double[][] {{ 1 },  { 1 }});
        Matrix B = matrix().of(new double[][] {{ 2, 3, 4, 5 }});

        double[][] result = matrix().values(matrix().multiply(A, B));
        assertThat(result).isEqualTo(new double[][] {
                { 2, 3, 4, 5 },
                { 2, 3, 4, 5 }
        });
    }

    @Test
    public void addMatrices() {
        Matrix A = matrix().of(new double[][] {
            { 10, 20 },
            { 30, 40 },
            { 50, 60 },
        });
        Matrix B = matrix().of(new double[][] {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 },
        });

        double[][] result = matrix().values(matrix().subtract(A, B));
        assertThat(result).isEqualTo(new double[][] {
            {  9, 18 },
            { 27, 36 },
            { 45, 54 },
        });
    }

    @Test
    public void shouldDivideSmallMatrix() {
        Matrix A = matrix().of(new double[][] {{ 2*3*5 }});
        Matrix B = matrix().of(new double[][] {{ 2, 3, 5 }});
        double[][] result = matrix().values(matrix().divide(A, B));
        assertThat(result).isEqualTo(new double[][] { { 3*5, 2*5, 2*3 }});
    }

    @Test
    public void shouldDivideTwoRowsMatrix() {
        Matrix A = matrix().of(new double[][] {{ 2*3*5 }, { 20*30*50}});
        Matrix B = matrix().of(new double[][] {
            { 2, 3, 5 }
        });
        double[][] result = matrix().values(matrix().divide(A, B));
        assertThat(result).isEqualTo(new double[][] {
            { 3*5, 2*5, 2*3 },
            { 30*10*50, 20*10*50, 20*10*30 }
        });
    }

    @Test
    public void shouldDivideLargeMatrix() {
        Matrix A = matrix().of(new double[][] {
            { 2*3*5 },
            { 20*30*50 },
            { 200*300*500 }
        }); // A[3x1]
        Matrix B = matrix().of(new double[][] {
            { 2, 3 },
            { 2, 5 },
            { 3, 5 },
        }); // B[3x2]
        double[][] result = matrix().values(matrix().divide(A, B));
        assertThat(result).isEqualTo(new double[][] {
            { 3*5, 2*5 },
            { 10*30*50, 20*30*10 },
            { 200*100*500, 200*300*100 }
        });
    }


    protected abstract MatrixOperations matrix();

}
