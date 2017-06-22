package com.johannesbrodwall.experimental.ml;

import java.io.IOException;
import java.text.DecimalFormat;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.eaxy.Content;
import org.eaxy.Element;
import org.eaxy.Node;
import org.eaxy.Xml;
import org.eaxy.html.Xhtml;

public class XorTrainingServlet extends HttpServlet {

    private MatrixOperations matrix = new SimpleMatrixOperations();
    private MatrixOperations.Function sigmoid = new MatrixOperations.Function() {
        @Override
        public double apply(double value) {
            return 1 / (1 + Math.exp(-value));
        }
    };
    private MatrixOperations.Function sigmoidPrime = new MatrixOperations.Function() {
        @Override
        public double apply(double value) {
            double y = sigmoid.apply(value);
            return y * (1.0 - y);
        }
    };

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        Xhtml xhtml = Xhtml.readAndClose(getClass().getResourceAsStream("/webapp-neural/xor-training.html.eaxy"));
        updateDocument(xhtml);
        resp.setContentType("text/html");
        xhtml.writeTo(resp.getWriter());
    }

    private void updateDocument(Xhtml xhtml) {
        Matrix inputMatrix = matrix.of(new double[][] {
            { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }
        });

        Matrix inputWeights = matrix.of(new double[][] {
            { 0.8, 0.4, 0.3 },
            {0.2, 0.9, 0.5}
        });
        Matrix outputWeights = matrix.of(new double[][] {
            { 0.3 }, { 0.5 }, { 0.9 }
        });

        Matrix outputTarget = matrix.of(new double[][] {
            { 0 }, { 1 }, { 1 }, { 0 }
        });

        Matrix hiddenValues = matrix.multiply(inputMatrix, inputWeights);
        Matrix hiddenActivations = matrix.apply(hiddenValues, sigmoid);
        Matrix outputValues = matrix.multiply(hiddenActivations, outputWeights);
        Matrix outputActivations = matrix.apply(outputValues, sigmoid);
        Matrix deltaOutput = matrix.subtract(outputTarget, outputActivations);
        Matrix outputPrime = matrix.apply(outputValues, sigmoidPrime);
        Matrix deltaOutputSum = matrix.multiplyElements(deltaOutput, outputPrime);
        Matrix deltaHiddenWeights = matrix.divide(deltaOutputSum, hiddenActivations);

        Matrix hiddenPrime = matrix.apply(hiddenValues, sigmoidPrime);
        Matrix deltaHidden = matrix.divideTransposed(deltaOutputSum, outputWeights);
        Matrix deltaHiddenSum = matrix.multiplyElements(deltaHidden, hiddenPrime);
        Matrix inputT = matrix.transpose(inputMatrix);
        Matrix deltaInputWeights = matrix.multiply(inputT, deltaHiddenSum);

        xhtml.findById("equation-1")
            .add(equation1(inputMatrix, inputWeights, hiddenValues));
        xhtml.findById("equation-2")
            .add(equation2(hiddenValues, hiddenActivations));
        xhtml.findById("equation-3")
            .add(equation3(hiddenActivations, outputWeights, outputValues));
        xhtml.findById("equation-4")
            .add(equation2(outputValues, outputActivations));
        xhtml.findById("equation-5")
            .add(equation5(outputValues, outputPrime, outputTarget, outputActivations, deltaOutput, deltaOutputSum));
        xhtml.findById("equation-6")
            .add(equation6(deltaOutputSum, hiddenActivations, deltaHiddenWeights));
        xhtml.findById("equation-7")
            .add(equation7(deltaOutputSum, outputWeights, hiddenValues, deltaHidden, hiddenPrime, deltaHiddenSum));
        xhtml.findById("equation-8")
            .add(equation8(inputT, deltaHiddenSum, deltaInputWeights));

        Matrix adjustedInputWeights = matrix.add(inputWeights, deltaInputWeights);

        //Matrix adjustedOutputWeights = matrix.add(outputWeights, matrix.transpose(deltaHiddenWeights));

        xhtml.findById("equation-9")
            .add(equation9(inputWeights, deltaInputWeights, adjustedInputWeights, outputWeights, matrix.transpose(deltaHiddenWeights), adjustedInputWeights));

    }

    private Node equation9(Matrix inputWeights, Matrix deltaInputWeights, Matrix adjustedInputWeights,
            Matrix outputWeights, Matrix deltaHiddenWeights, Matrix adjustedOutputWeights) {
        return Xml.el("table", Xml.attr("class", "equation"),
                Xml.el("tr",
                    Xml.el("td", matrixToHtml(inputWeights)),
                    Xml.el("td", Xml.text("-")),
                    Xml.el("td", matrixToHtml(deltaInputWeights)),
                    Xml.el("td", Xml.text("=")),
                    Xml.el("td", matrixToHtml(adjustedInputWeights))),
                Xml.el("tr",
                    Xml.el("td", "weights"),
                    Xml.el("td"),
                    Xml.el("td", "adjustment"),
                    Xml.el("td"),
                    Xml.el("td", "new weights"),
                Xml.el("tr",
                    Xml.el("td", matrixToHtml(outputWeights)),
                    Xml.el("td", Xml.text("-")),
                    Xml.el("td", matrixToHtml(deltaHiddenWeights)),
                    Xml.el("td", Xml.text("=")),
                    Xml.el("td", matrixToHtml(adjustedOutputWeights)))));
    }

    private Node equation8(Matrix inputMatrix, Matrix deltaHiddenSum, Matrix deltaInputWeights) {
        return Xml.el("table", Xml.attr("class", "equation"),
                Xml.el("tr",
                    Xml.el("td", matrixToHtml(inputMatrix)),
                    Xml.el("td", Xml.text("*")),
                    Xml.el("td", matrixToHtml(deltaHiddenSum)),
                    Xml.el("td", Xml.text("=")),
                    Xml.el("td", matrixToHtml(deltaInputWeights))),
                Xml.el("tr",
                    Xml.el("td", "transposed(input)"),
                    Xml.el("td"),
                    Xml.el("td", "delta sum"),
                    Xml.el("td"),
                    Xml.el("td", "delta weights")));
    }

    private Node equation7(Matrix deltaOutput, Matrix outputWeights, Matrix hiddenValues, Matrix deltaHidden, Matrix hiddenPrime, Matrix deltaHiddenSum) {
        return Xml.el("table", Xml.attr("class", "equation"),
                Xml.el("tr",
                    Xml.el("td", Xml.text("S'(sum) * (target - output)")),
                    Xml.el("td", Xml.text("=")),
                    Xml.el("td", matrixToHtml(hiddenPrime)),
                    Xml.el("td", Xml.text(" ⊙ (")),
                    Xml.el("td", matrixToHtml(deltaOutput)),
                    Xml.el("td", Xml.text("/")),
                    Xml.el("td", matrixToHtml(outputWeights)),
                    Xml.el("td", Xml.text(") =")),
                    Xml.el("td", matrixToHtml(hiddenPrime)),
                    Xml.el("td", Xml.text("⊙")),
                    Xml.el("td", matrixToHtml(deltaHidden)),
                    Xml.el("td", Xml.text("=")),
                    Xml.el("td", matrixToHtml(deltaHiddenSum))));
    }

    private Node equation6(Matrix deltaOutputSum, Matrix hiddenActivations, Matrix deltaHiddenWeights) {
        return Xml.el("table", Xml.attr("class", "equation"),
                Xml.el("tr",
                        Xml.el("td", matrixToHtml(deltaOutputSum)),
                        Xml.el("td", Xml.text("/")),
                        Xml.el("td", matrixToHtml(hiddenActivations)),
                        Xml.el("td", Xml.text("=")),
                        Xml.el("td", matrixToHtml(deltaHiddenWeights))),
                Xml.el("tr",
                        Xml.el("td", "Delta sum"),
                        Xml.el("td"),
                        Xml.el("td", "Activations"),
                        Xml.el("td"),
                        Xml.el("td", "Weights"))
                );
    }

    private Node equation5(Matrix outputValues, Matrix valuesPrime, Matrix outputTarget, Matrix outputActivations, Matrix delta, Matrix deltaOutputSum) {
        return Xml.el("table", Xml.attr("class", "equation"),
                Xml.el("tr",
                        Xml.el("td", Xml.text("S'(sum) ⊙ (target - output)")),
                        Xml.el("td", Xml.text("=")),
                        Xml.el("td", Xml.text("S'(")),
                        Xml.el("td", matrixToHtml(outputValues)),
                        Xml.el("td", Xml.text(") ⊙ (")),
                        Xml.el("td", matrixToHtml(outputTarget)),
                        Xml.el("td", Xml.text("-")),
                        Xml.el("td", matrixToHtml(outputActivations)),
                        Xml.el("td", Xml.text(") =")),
                        Xml.el("td", matrixToHtml(valuesPrime)),
                        Xml.el("td", Xml.text("⊙")),
                        Xml.el("td", matrixToHtml(delta)),
                        Xml.el("td", Xml.text("=")),
                        Xml.el("td", matrixToHtml(deltaOutputSum))))
                ;
    }

    private Node equation3(Matrix hiddenActivations, Matrix outputWeights, Matrix outputValues) {
        return Xml.el("table", Xml.attr("class", "equation"),
            Xml.el("tr",
                    Xml.el("td", matrixToHtml(hiddenActivations)),
                    Xml.el("td", Xml.text("×")),
                    Xml.el("td", matrixToHtml(outputWeights)),
                    Xml.el("td", Xml.text("=")),
                    Xml.el("td", matrixToHtml(outputValues))),
            Xml.el("tr",
                    Xml.el("td", "Hidden activation"),
                    Xml.el("td"),
                    Xml.el("td", "Weight"),
                    Xml.el("td"),
                    Xml.el("td", "Output values")
                    ));
    }

    private Node equation2(Matrix hiddenValues, Matrix hiddenActivations) {
        return Xml.el("table", Xml.attr("class", "equation"),
                Xml.el("tr",
                        Xml.el("td", "Sigmoid("),
                        Xml.el("td", matrixToHtml(hiddenValues)),
                        Xml.el("td", Xml.text(") =")),
                        Xml.el("td", matrixToHtml(hiddenActivations))),
                Xml.el("tr",
                        Xml.el("td"),
                        Xml.el("td", "Hidden values"),
                        Xml.el("td"),
                        Xml.el("td", "Hidden activations")
                        ));
    }

    private Node equation1(Matrix inputMatrix, Matrix inputWeights, Matrix hiddenValues) {
        return Xml.el("table", Xml.attr("class", "equation"),
                Xml.el("tr",
                        Xml.el("td", matrixToHtml(inputMatrix)),
                        Xml.el("td", Xml.text("×")),
                        Xml.el("td", matrixToHtml(inputWeights)),
                        Xml.el("td", Xml.text("=")),
                        Xml.el("td", matrixToHtml(hiddenValues))),
                Xml.el("tr",
                        Xml.el("td", "All inputs"),
                        Xml.el("td"),
                        Xml.el("td", "Weight"),
                        Xml.el("td"),
                        Xml.el("td", "Hidden values")
                        ));
    }

    private Content matrixToHtml(Matrix m) {
        DecimalFormat formatText = new DecimalFormat("#.##");
        DecimalFormat formatTitle = new DecimalFormat("#.######");
        double[][] values = matrix.values(m);
        Element el = Xml.el("table", Xml.attr("class", "matrix"));
        for (int row = 0; row < values.length; row++) {
            Element rowEl = Xml.el("tr");
            for (int col = 0; col < values[row].length; col++) {
                rowEl.add(Xml.el("td",
                        Xml.attr("title", formatTitle.format(values[row][col])),
                        Xml.text(formatText.format(values[row][col]))));
            }
            el.add(rowEl);
        }
        return el;
    }
}
