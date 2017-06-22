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

public class XorPresetServlet extends HttpServlet {

    private MatrixOperations matrix = new SimpleMatrixOperations();
    private MatrixOperations.Function sigmoid = new MatrixOperations.Function() {
        @Override
        public double apply(double value) {
            return 1 / (1 + Math.exp(-value));
        }
    };

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        Xhtml xhtml = Xhtml.readAndClose(getClass().getResourceAsStream("/webapp-neural/xor-presets.html.eaxy"));
        updateDocument(xhtml);
        resp.setContentType("text/html");
        xhtml.writeTo(resp.getWriter());
    }

    private void updateDocument(Xhtml xhtml) {
        Matrix inputMatrix = matrix.of(new double[][] {
            { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }
        });

        Matrix inputWeights = matrix.of(new double[][] {
                { 54, 17 }, { 14, 14 }
        });
        Matrix inputBiases = matrix.of(new double[][] {{ -8 , -20 }});
        Matrix outputWeights = matrix.of(new double[][] {
            { 92 }, { -98 }
        });
        Matrix outputBiases = matrix.ofTransposed(new double[][] {{ -48 }});

        Matrix hiddenValues = matrix.multiplyAdd(inputMatrix, inputWeights, inputBiases);
        Matrix hiddenActivations = matrix.apply(hiddenValues, sigmoid);
        Matrix outputValues = matrix.multiplyAdd(hiddenActivations, outputWeights, outputBiases);
        Matrix outputActivations = matrix.apply(outputValues, sigmoid);

        xhtml.findById("equation-1")
            .add(equation1(inputMatrix, inputWeights, inputBiases, hiddenValues));
        xhtml.findById("equation-2")
            .add(equation2(hiddenValues, hiddenActivations));
        xhtml.findById("equation-3")
            .add(equation3(hiddenActivations, outputWeights, outputBiases, outputValues));
        xhtml.findById("equation-4")
            .add(equation2(outputValues, outputActivations));
    }

    private Node equation3(Matrix hiddenActivations, Matrix outputWeights, Matrix outputBiases, Matrix outputValues) {
        return Xml.el("table", Xml.attr("class", "equation"),
            Xml.el("tr",
                    Xml.el("td", matrixToHtml(hiddenActivations)),
                    Xml.el("td", Xml.text("×")),
                    Xml.el("td", matrixToHtml(outputWeights)),
                    Xml.el("td", Xml.text("+")),
                    Xml.el("td", matrixToHtml(outputBiases)),
                    Xml.el("td", Xml.text("=")),
                    Xml.el("td", matrixToHtml(outputValues))),
            Xml.el("tr",
                    Xml.el("td", "Hidden activation"),
                    Xml.el("td"),
                    Xml.el("td", "Weight"),
                    Xml.el("td"),
                    Xml.el("td", "Biases"),
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

    private Node equation1(Matrix inputMatrix, Matrix inputWeights, Matrix inputBiases, Matrix hiddenValues) {
        return Xml.el("table", Xml.attr("class", "equation"),
                Xml.el("tr",
                        Xml.el("td", matrixToHtml(inputMatrix)),
                        Xml.el("td", Xml.text("×")),
                        Xml.el("td", matrixToHtml(inputWeights)),
                        Xml.el("td", Xml.text("+")),
                        Xml.el("td", matrixToHtml(inputBiases)),
                        Xml.el("td", Xml.text("=")),
                        Xml.el("td", matrixToHtml(hiddenValues))),
                Xml.el("tr",
                        Xml.el("td", "All inputs"),
                        Xml.el("td"),
                        Xml.el("td", "Weight"),
                        Xml.el("td"),
                        Xml.el("td", "Biases"),
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
