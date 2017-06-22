package com.johannesbrodwall.experimental.ml;

import com.johannesbrodwall.experimental.ml.MatrixOperations;
import com.johannesbrodwall.experimental.ml.SimpleMatrixOperations;

public class SimpleMatrixTest extends MatrixTest {

    private MatrixOperations operations = new SimpleMatrixOperations();

    @Override
    protected MatrixOperations matrix() {
        return operations;
    }

}
