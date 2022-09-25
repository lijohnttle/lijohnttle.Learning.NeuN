using MathNet.Numerics.LinearAlgebra;

namespace lijohnttle.Learning.NeuN.Core.Internal
{
    internal class NetworkTrainer : INetworkTrainer
    {
        public static readonly NetworkTrainer Default = new NetworkTrainer();


        public Matrix<double>[] Train(
            Vector<double> inputs,
            Vector<double> targets,
            Matrix<double>[] weightsLayers,
            double learningRate,
            Func<double, double> activationFunction,
            IQueryProcessor queryProcessor)
        {
            var newWeightsLayers = (Matrix<double>[])weightsLayers.Clone();
            
            var outputs = queryProcessor.Query(inputs, weightsLayers, activationFunction).ToList();
            outputs.Insert(0, inputs);

            var finalOutputs = outputs.Last();
            var finalErrors = targets - finalOutputs;
            
            var previousErrros = finalErrors;

            for (int i = weightsLayers.Length - 1; i >= 0; i--)
            {
                var currentLayerOutput = outputs[i + 1];
                var previousLayerOutput = outputs[i];

                var errors = i == weightsLayers.Length - 1
                    ? finalErrors
                    : weightsLayers[i + 1].Transpose() * previousErrros;

                var weightsDelta = ComputeWeightsDelta(learningRate, errors, currentLayerOutput, previousLayerOutput);

                newWeightsLayers[i] += weightsDelta;

                previousErrros = errors;
            }

            return newWeightsLayers;
        }


        private Matrix<double> ComputeWeightsDelta(
            double learningRate,
            Vector<double> errors,
            Vector<double> currentLayerOutput,
            Vector<double> previousLayerOutput)
        {
            var M = Matrix<double>.Build;

            var sigmoidComponent = errors.PointwiseMultiply(currentLayerOutput.PointwiseMultiply(1 - currentLayerOutput));
            
            return learningRate * M.DenseOfColumnVectors(sigmoidComponent) * M.DenseOfRowVectors(previousLayerOutput);
        }
    }
}
