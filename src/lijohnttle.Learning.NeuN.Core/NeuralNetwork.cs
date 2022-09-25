using lijohnttle.Learning.NeuN.Core.Internal;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;

namespace lijohnttle.Learning.NeuN.Core
{
    public class NeuralNetwork : INeuralNetwork
    {
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

            weightsLayers = new[]
            {
                CreateWeightsMatrix(hiddenLayerSize, inputLayerSize),   // input to hidden layer
                CreateWeightsMatrix(outputLayerSize, hiddenLayerSize)   // hidden to output layer
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

            this.weightsLayers = this.networkTrainer
                .Train(
                    V.DenseOfArray(inputs),
                    V.DenseOfArray(targets),
                    this.weightsLayers,
                    this.LearningRate,
                    this.activationFunction,
                    this.queryProcessor);
        }

        public double[] Query(double[] inputs)
        {
            if (inputs.Length != InputLayerSize)
            {
                throw new InvalidOperationException($"Inputs vector does not have the required size of {InputLayerSize}");
            }

            var V = Vector<double>.Build;

            return this.queryProcessor
                .Query(
                    V.DenseOfArray(inputs),
                    this.weightsLayers,
                    this.activationFunction)
                .Last()
                .ToArray();
        }


        private static Matrix<double> CreateWeightsMatrix(int rows, int columns)
        {
            var M = Matrix<double>.Build;

            return M.Random(rows, columns,
                new Normal(0, Math.Pow(rows, -0.5), SystemRandomSource.Default));
        }
    }
}