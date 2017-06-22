I'm having a little fun implementing matrix operations and machine learning in Java.

Implements an artificial neural network feed forward and backpropagation algorithm,
along with web-based visualizations.

The backpropagation algorithm seems to be buggy!

Prerequisites:
--------------

* Java 8
* Maven
* IntelliJ or Eclipse

Getting started:
----------------

1. Run `mvn eclipse:eclipse` and import the project in Eclipse (File -> Import -> Existing Projects into workspace)
  * Alternatively: Import the maven project in IntelliJ
  * Alternatively: Eclipse: File -> Import -> Existing Maven Projects
2. Run the tests (only the matrix operations have test)
3. Run the main class `com.johannesbrodwall.experimental.ml.NeuralServer`
4. Explore the examples:
   * http://localhost:8888/index.html demonstrates and visualizes matrix operations. Implementation in `com.johannesbrodwall.experimental.ml.NeuralServlet`
   * http://localhost:8888/xor-presets demonstrates and visualizes the calculations to recognize XOR with some preset biases. Implementation in `com.johannesbrodwall.experimental.ml.XorPresetServlet`
   * http://localhost:8888/xor-training demonstrates and visualizes the calculations for backpropagation for XOR, but it's not working correctly! Implementation in `com.johannesbrodwall.experimental.ml.XorTrainingServlet`

