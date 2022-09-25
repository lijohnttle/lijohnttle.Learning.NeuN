using lijohnttle.Learning.NeuN.Core.Internal;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;

namespace lijohnttle.Learning.NeuN.Core
{
    public class NeuralNetwork
    {
        private readonly Matrix<double> inputToHiddenWeights;
        private readonly Matrix<double> hiddenToOutputWeights;
        private readonly IQueryProcessor queryProcessor = QueryProcessor.Default;
        private readonly INetworkTrainer networkTrainer = NetworkTrainer.Default;
        private readonly Func<double, double> activationFunction = SpecialFunctions.Logistic;
        private Matrix<double>[] weightsLayers;


        public NeuralNetwork(
            int inputLayerSize,
            int hiddenLayerSize,
            int outputLayerSize,
            double learningRate)
        {
            InputLayerSize = inputLayerSize;
            HiddenLayerSize = hiddenLayerSize;
            OutputLayerSize = outputLayerSize;
            LearningRate = learningRate;

            var M = Matrix<double>.Build;

            inputToHiddenWeights = M.Random(hiddenLayerSize, inputLayerSize,
                new Normal(0, Math.Pow(hiddenLayerSize, -0.5), SystemRandomSource.Default));
            hiddenToOutputWeights = M.Random(outputLayerSize, hiddenLayerSize,
                new Normal(0, Math.Pow(outputLayerSize, -0.5), SystemRandomSource.Default));

            weightsLayers = new[]
            {
                inputToHiddenWeights,
                hiddenToOutputWeights
            };
        }


        public int InputLayerSize { get; }

        public int HiddenLayerSize { get; }

        public int OutputLayerSize { get; }

        public double LearningRate { get; }


        public void Train(double[] inputs, double[] targets)
        {
            if (inputs.Length != InputLayerSize)
            {
                throw new InvalidOperationException($"Inputs vector does not have the required size of {InputLayerSize}");
            }

            if (targets.Length != OutputLayerSize)
            {
                throw new InvalidOperationException($"Targets vector does not have the required size of {OutputLayerSize}");
            }
            
            var V = Vector<double>.Build;

            var inputsVector = V.DenseOfArray(inputs);
            var targetsVector = V.DenseOfArray(targets);

            this.weightsLayers = this.networkTrainer
                .Train(inputsVector, targetsVector, this.weightsLayers, this.LearningRate, this.activationFunction, this.queryProcessor);
        }

        public double[] Query(double[] inputs)
        {
            if (inputs.Length != InputLayerSize)
            {
                throw new InvalidOperationException($"Inputs vector does not have the required size of {InputLayerSize}");
            }

            var V = Vector<double>.Build;

            var inputsVector = V.DenseOfArray(inputs);

            return this.queryProcessor
                .Query(inputsVector, this.weightsLayers, this.activationFunction)
                .Last()
                .ToArray();
        }
    }
}