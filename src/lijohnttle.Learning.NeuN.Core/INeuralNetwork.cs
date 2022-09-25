namespace lijohnttle.Learning.NeuN.Core
{
    public interface INeuralNetwork
    {
        public int InputLayerSize { get; }

        public int HiddenLayerSize { get; }

        public int OutputLayerSize { get; }

        public double LearningRate { get; }


        /// <summary>
        /// Trains a neural network.
        /// </summary>
        /// <param name="inputs">Array with input values.</param>
        /// <param name="targets">Array with target values that is used to calculate errors.</param>
        void Train(double[] inputs, double[] targets);

        /// <summary>
        /// Processes an input signal.
        /// </summary>
        /// <param name="inputs">Array with input values.</param>
        /// <returns>Array with output values.</returns>
        double[] Query(double[] inputs);
    }
}
