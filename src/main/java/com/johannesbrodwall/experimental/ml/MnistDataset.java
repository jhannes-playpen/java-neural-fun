package com.johannesbrodwall.experimental.ml;

import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.io.Closeable;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import javax.imageio.ImageIO;

import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;

public class MnistDataset implements Closeable {

    private DataInputStream imageStream;
    private DataInputStream labelStream;
    private byte[] image;
    private int rows;
    private int cols;
    private int imageCount;

    public MnistDataset(String imageFile, String labelFile) throws IOException {
        imageStream = gzipDataStream(imageFile);
        labelStream = gzipDataStream(labelFile);

        int magic = imageStream.readInt();
        if (magic != 2051) {
            throw new IllegalArgumentException("Invalid magic in dataset");
        }
        int labelMagic = labelStream.readInt();
        if (labelMagic != 2049) {
            throw new IllegalArgumentException("Invalid magic in dataset");
        }

        imageCount = imageStream.readInt();
        int labelCount = labelStream.readInt();
        if (imageCount != labelCount) {
            throw new IllegalArgumentException(imageCount + " images, but " + labelCount + " labels");
        }

        rows = imageStream.readInt();
        cols = imageStream.readInt();
        image = new byte[rows*cols];
    }

    public static void main(String[] args) throws IOException {
        try(MnistDataset extractDigits = new MnistDataset("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")) {
            extractDigits.writeImages();

            System.out.println(extractDigits.labelsAsInt(10));
        }
    }

    private void writeImages() throws IOException {
        new File("tmp").mkdirs();
        for (int i=0; i<100; i++) {
            imageStream.readFully(image);

            int label = labelStream.read();
            String filename = "tmp/image-" + label + ".png";
            ImageIO.write(grayscaleBmp(image, rows, cols), "png", new File(filename));
            System.out.println("Wrote file " + filename);
        }
    }

    private static DataInputStream gzipDataStream(String filename) throws IOException, FileNotFoundException {
        return new DataInputStream(new GZIPInputStream(new FileInputStream(filename)));
    }

    private static BufferedImage grayscaleBmp(byte[] image, int rows, int cols) {
        ColorModel cm = new ComponentColorModel(ColorSpace.getInstance(ColorSpace.CS_GRAY), new int[] { 8 },
                false, true, Transparency.OPAQUE, DataBuffer.TYPE_BYTE);
        SampleModel sm = cm.createCompatibleSampleModel(cols, rows);
        WritableRaster raster = Raster.createWritableRaster(sm, new DataBufferByte(image, cols * rows), null);
        BufferedImage renderedImage = new BufferedImage(cm, raster, false, null);
        return renderedImage;
    }

    @Override
    public void close() throws IOException {
        imageStream.close();
        labelStream.close();
    }

    public List<Vector> readImages() throws IOException {
        return readImages(imageCount);
    }

    private List<Vector> readImages(int count) throws IOException {
        List<Vector> result = new ArrayList<>();
        for (int i=0; i<count; i++) {
            imageStream.readFully(image);
            result.add(toVector32f(this.image));
        }
        return result;
    }

    private DenseVector toVector32f(byte[] image) {
        double[] image32f = new double[image.length];
        for (int j = 0; j < image32f.length; j++) {
            image32f[j] = image[j]/256.0;
        }
        return new DenseVector(image32f, false);
    }

    public List<Integer> labelsAsInt(int count) throws IOException {
        List<Integer> result = new ArrayList<>();
        byte[] labels = new byte[count];
        labelStream.readFully(labels);
        for (int i=0; i<count; i++) {
            result.add((int)labels[i]);
        }
        return result;
    }

    public List<Integer> labelsAsInt() throws IOException {
        return labelsAsInt(imageCount);
    }

    public List<Vector> readLabelsInverseActivations() throws IOException {
        return readLabelsInverseActivations(imageCount);
    }

    private List<Vector> readLabelsInverseActivations(int count) throws IOException {
        List<Vector> result = new ArrayList<>();
        byte[] labels = new byte[count];
        labelStream.readFully(labels);
        for (int i=0; i<count; i++) {
            result.add(toInverseActivation(labels[i]));
        }
        return result;
    }

    private Vector toInverseActivation(int value) {
        double[] activation = new double[10];
        for (int j = 0; j < activation.length; j++) {
            activation[j] = 1.0;
        }
        activation[value] = 0.0;
        return new DenseVector(activation, false);
    }

    public List<Integer> readLabels() throws IOException {
        List<Integer> result = new ArrayList<>();
        byte[] labels = new byte[imageCount];
        labelStream.readFully(labels);
        for (int i=0; i<imageCount; i++) {
            result.add((int)labels[i]);
        }
        return result;
    }

}
