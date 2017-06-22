package com.johannesbrodwall.experimental.ml;

import org.eclipse.jetty.server.Handler;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.util.resource.Resource;
import org.eclipse.jetty.webapp.WebAppContext;

public class NeuralServer {

    private Server server;

    public NeuralServer(int port) {
        this.server = new Server(port);
    }

    public static void main(String[] args) throws Exception {
        NeuralServer server = new NeuralServer(8888);
        server.create();
        server.start();
    }

    private void create() {
        server.setHandler(createHandler());
    }

    private Handler createHandler() {
        WebAppContext handler = new WebAppContext();
        handler.setContextPath("/");
        handler.setBaseResource(Resource.newClassPathResource("webapp-neural"));
        handler.addServlet(new ServletHolder(new XorPresetServlet()), "/xor-presets");
        handler.addServlet(new ServletHolder(new XorTrainingServlet()), "/xor-training");
        handler.addServlet(new ServletHolder(new NeuralServlet()), "/index.html");
        return handler;
    }

    private void start() throws Exception {
        server.start();
    }

}
