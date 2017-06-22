package com.johannesbrodwall.experimental.ml;

import com.johannesbrodwall.experimental.ml.CiprMatrixOperations;
import com.johannesbrodwall.experimental.ml.MatrixOperations;

public class CiprMatrixTest extends MatrixTest {

    private MatrixOperations operations = new CiprMatrixOperations();

    @Override
    protected MatrixOperations matrix() {
        return operations;
    }

}
