package com.johannesbrodwall.experimental.ml;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.eaxy.Content;
import org.eaxy.Element;
import org.eaxy.Xml;

public class NeuralServlet extends HttpServlet {

    private double matrix1[][] = {
            { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }
    };

    private double matrix2[][] = {
            { 0.8, 0.4, 0.3 }, { 0.2, 0.9, 0.5 }
    };
    private double matrix3[][] = {
            { 0.1 ,  0.4,  0.3 }
    };
    private double matrix4[][] = {
            { 0, 0, 0 }, { 0.2, 0.9, 0.5 }, { 0.8, 0.4, 0.3 }, { 1.0, 1.3, 0.8 }
    };

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        resp.setContentType("text/html");
        resp.getWriter().write(createHtml(createOperation()).toXML());
    }

    private Element createOperation() {
        return Xml.el("table", Xml.attr("class", "equation"),
                Xml.el("tr",
                        Xml.el("td", matrixToHtml(matrix1)),
                        Xml.el("td", Xml.text("×")),
                        Xml.el("td", matrixToHtml(matrix2)),
                        Xml.el("td", Xml.text("+")),
                        Xml.el("td", matrixToHtml(matrix3)),
                        Xml.el("td", Xml.text("=")),
                        Xml.el("td", matrixToHtml(matrix4))));
    }

    private Content matrixToHtml(double[][] matrix) {
        Element el = Xml.el("table", Xml.attr("class", "matrix"));
        for (int row = 0; row < matrix.length; row++) {
            Element rowEl = Xml.el("tr");
            for (int col = 0; col < matrix[row].length; col++) {
                String d = String.valueOf(matrix[row][col]);
                rowEl.add(Xml.el("td", Xml.text(d)));
            }
            el.add(rowEl);
        }
        return el;
    }

    private Element createHtml(Element operation) {
        return Xml.el("html",
                Xml.el("head",
                        Xml.el("link",
                                Xml.attr("rel", "stylesheet"),
                                Xml.attr("type", "text/css"),
                                Xml.attr("href", "style/neural.css"))),
                Xml.el("body", operation));
    }

}
